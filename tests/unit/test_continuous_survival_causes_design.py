"""The declaration of ``continuous-survival-causes`` (X6 plan rev 5, section 5.3)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cleverly.survival import result as survival_result
from tests.studies import continuous_survival_causes as study
from tests.studies import continuous_survival_causes_law as laws
from tests.studies import continuous_survival_causes_pilot as pilot
from tests.studies import continuous_survival_causes_properties as properties
from tests.studies import continuous_survival_law as base
from tests.studies import repeated_sizing as sizing
from tests.studies.evidence.seeds import replicate_seed, stream_seed

TESTS = Path(__file__).resolve().parents[1]
PILOT = pilot.read()
PERFORMANCE = study.STUDY.artifacts / "performance-tests.csv"
PROPERTIES = study.STUDY.artifacts / "properties.csv"


def test_the_declaration_is_the_plans() -> None:
    assert study.STUDY.replicates == study.PRIMARY_REPLICATES
    assert study.REPLICATE_CAP == 7_600
    assert study.PILOT_REPLICATES == 800
    assert study.MINIMUM_REPLICATES == 1_000
    assert study.PRIMARY_N == properties.ROBUSTNESS_N == properties.CALIBRATION_N == 1_000
    assert properties.ROBUSTNESS_REPLICATES == 1_600  # design review b, S2
    assert properties.SHRUNKEN_SE_FACTOR == 0.85
    assert study.MARGINS.standardized_bias == 0.2
    assert study.STUDY.reference is None
    assert study.FAILURE_BOUND == 0.01
    assert study.CELLS == {"l1_cf_1k": "one_cause", "l3_cf_1k": "three_causes"}
    assert laws.CAUSE3 == (-2.5, 0.5, 0.5, 0.5)
    assert {law.causes for law in laws.LAWS.values()} == {1, 3}
    # The share floor part (a) measured is 0.40, so the first quartile is not licensed.
    assert survival_result.MEASURED_EVENT_SHARE[0] == survival_result.SHARE_FLOOR_DEFAULT == 0.4
    # The declared property cells are the ones the module writes.
    declared = {
        (family, cell) for family, cells in study.STUDY.property_cells.items() for cell in cells
    }
    assert declared == {(family, cell) for family, cell, _, _ in properties.declared_truths()}


def test_each_cell_gates_its_median_and_upper_quartile() -> None:
    expected = {"l1_cf_1k": (6, 3), "l3_cf_1k": (18, 9)}
    for cell, law in study.CELLS.items():
        gated, early = laws.gated(law), laws.first_quartile(law)
        assert (len(gated), len(early)) == expected[cell]
        assert study.ESTIMANDS[cell] == gated
        assert not set(gated) & set(early)
        assert all(f"t={laws.LAWS[law].times[0]:g}]" in name for name in early)
    assert study.EARLY_COUNT == 9


def test_the_seeds_are_used_nowhere_else_in_tests() -> None:
    mine = {str(study.SEED), str(study.RESAMPLING_SEED)}
    for path in TESTS.rglob("*.py"):
        if path.name in {"continuous_survival_causes.py", Path(__file__).name}:
            continue
        numbers = set(re.findall(r"\d{7,}", path.read_text(encoding="utf-8").replace("_", "")))
        assert not mine & numbers, path


def _declared_streams() -> dict[str, list[int]]:
    """Every seed a declared stream reads, at its declared count."""
    streams: dict[str, list[int]] = {}
    for cell in study.CELLS:
        streams[f"primary/{cell}"] = [
            replicate_seed(study.STUDY, cell, r) for r in range(study.PRIMARY_REPLICATES)
        ]
        streams[f"pilot/{cell}"] = [pilot.pilot_seed(cell, r) for r in range(pilot.EXTENSION_DRAWS)]
    for law in laws.LAWS:
        streams[f"pilot/{law}_controls"] = [
            pilot.pilot_seed(f"{law}_controls", r) for r in range(pilot.CONTROL_DRAWS)
        ]
        streams[f"robust/{law}"] = [
            stream_seed(study.STUDY, "property_sample", properties.ROBUSTNESS, law, r)
            for r in range(properties.ROBUSTNESS_REPLICATES)
        ]
    for label in properties.CALIBRATION_LABELS:
        streams[f"calibration/{label}"] = [
            stream_seed(study.STUDY, "property_sample", properties.CALIBRATION, label, r)
            for r in range(properties.CALIBRATION_REPLICATES)
        ]
    return streams


def test_the_streams_are_disjoint() -> None:
    # At the declared counts, the primary draws (SEED), the pilots, the robustness and the
    # calibration draws (RESAMPLING_SEED) never share a seed, within a stream or across them.
    seen: set[int] = set()
    for name, seeds in _declared_streams().items():
        assert len(set(seeds)) == len(seeds), name
        assert not set(seeds) & seen, name
        seen |= set(seeds)


def test_the_laws_are_the_base_laws_with_the_causes_changed() -> None:
    x1, x2, x3 = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([-1.0, 0.5, 2.0])
    for z in (0.0, 1.0):
        h1, h2 = base.rates(x1, x2, x3, z)
        (one,) = laws.LAWS["one_cause"].rates(x1, x2, x3, z)
        a, b, c = laws.LAWS["three_causes"].rates(x1, x2, x3, z)
        np.testing.assert_array_equal(one, h1)
        np.testing.assert_array_equal(a, h1)
        np.testing.assert_array_equal(b, h2)
        np.testing.assert_allclose(c, np.exp(-2.5 + 0.5 * x1 + 0.5 * x3 + 0.5 * z))
    # Witness: the third cause competes, so the three-cause law's cause-1 risk falls below the
    # base law's at the same time, and L1's (no competition) lies above it.
    t = 1.12
    two = base.risk(1, 1, t)
    assert laws.LAWS["three_causes"].risk(1, 1, t) < two - 0.01
    assert laws.LAWS["one_cause"].risk(1, 1, t) > two + 0.005
    frame = laws.LAWS["one_cause"].sample(20_000, 1)
    assert set(frame["event"]) == {0, 1}
    frame = laws.LAWS["three_causes"].sample(20_000, 1)
    assert set(frame["event"]) == {0, 1, 2, 3}


def test_each_truth_is_the_counterfactual_risk_by_simulation() -> None:
    # 400,000 uncensored counterfactual draws per arm: every truth within 4 Monte Carlo SEs.
    rng = np.random.default_rng(20261119)
    for name, law in laws.LAWS.items():
        truth = laws.truth(name)
        n = 400_000
        factor = rng.integers(0, 3, n)
        x1, x2 = (factor == 1).astype(float), (factor == 2).astype(float)
        x3 = rng.normal(size=n)
        for arm, z in laws.ARMS.items():
            hazards = np.vstack(law.rates(x1, x2, x3, float(z)))
            total = hazards.sum(axis=0)
            event_time = rng.exponential(1.0 / total)
            cumulative = np.cumsum(hazards / total, axis=0)
            cause = 1 + (rng.uniform(size=n)[None, :] > cumulative[:-1]).sum(axis=0)
            for code in law.labels:
                for t in law.times:
                    hit = (event_time <= t) & (cause == code)
                    se = float(np.sqrt(hit.mean() * (1 - hit.mean()) / n))
                    assert abs(hit.mean() - truth[law.level_name(arm, code, t)]) < 4 * se


def test_the_times_are_each_laws_event_time_quartiles() -> None:
    assert laws.ONE_CAUSE_TIMES == (0.58, 1.48, 3.24)
    assert laws.THREE_CAUSE_TIMES == (0.37, 0.9, 1.86)
    for law in laws.LAWS.values():
        assert law.event_time_quartiles(n=400_000) == pytest.approx(law.times, abs=0.02)


def test_the_efficient_sd_reads_every_cause() -> None:
    # With three causes the martingale term carries h - h_j over all of them: dropping the
    # third cause from h (the two-cause closed form) moves the efficient SD of a cause-1 risk.
    law = laws.LAWS["three_causes"]
    sd = law.efficiency_sd(1, law.times[1])
    assert all(value > 0 for value in sd.values())
    two = base.efficiency_sd(1, law.times[1])
    assert abs(sd["treated"] - two["treated"]) > 0.01


def test_the_counts_are_the_rules_on_the_committed_pilot() -> None:
    assert not PILOT.empty
    for cell in study.CELLS:
        assert pilot.draws(PILOT, cell) == pilot.EXTENSION_DRAWS
    primary = pilot.primary_count(PILOT)
    calibration = pilot.calibration_count(PILOT)
    assert primary == study.PRIMARY_REPLICATES
    assert calibration == properties.CALIBRATION_REPLICATES
    assert study.MINIMUM_REPLICATES <= primary <= study.REPLICATE_CAP
    assert calibration <= study.REPLICATE_CAP
    # The shrunk-SE control leaves the band with certainty at the declared count.
    for power in pilot.shrunken_power(PILOT, calibration).values():
        assert power > 0.999


def test_the_robustness_count_passes_the_positive_cells() -> None:
    # Plan 5.3's 1,000 draws leave the three positive cells passing together with about 0.94; the
    # declared count lifts the plug-in family pass above 0.95, and every control keeps its power.
    assert pilot.robustness_pass(PILOT, 1_000) < sizing.FAMILY_WISE_PASS
    assert pilot.robustness_pass(PILOT, properties.ROBUSTNESS_REPLICATES) >= 0.99
    readings = pilot.control_readings(PILOT)
    assert (readings.loc[readings["role"] == "control", "pass_probability"] > 0.999).all()


def test_the_pilot_checks_the_failures_of_every_declared_cell() -> None:
    families = pilot._failure_families(study.PRIMARY_REPLICATES)
    assert len(families) == len(study.CELLS) + len(properties.CALIBRATION_LABELS) + len(laws.LAWS)
    assert pilot.failure_refusals(PILOT, study.PRIMARY_REPLICATES) == []
    # Nonzero witness: 20 of 400 control fits failed predicts a breach of the robustness bound.
    rows = PILOT.assign(refusal=PILOT["refusal"].astype(object))
    failed = (rows["label"] == "three_causes_controls") & (rows["replicate"] < 20)
    rows.loc[failed, "refusal"] = "DataError"
    refusals = pilot.failure_refusals(rows, study.PRIMARY_REPLICATES)
    assert len(refusals) == 1
    assert refusals[0].startswith("double_robustness/three_causes lost 20 of 400")


def test_the_at_target_count_and_the_shrunk_control() -> None:
    count = pilot.at_target_count()
    row = float(sizing.calibration_pass(study.MARGINS, count, np.array([1.0]), 0.95)[0])
    assert row**2 >= sizing.FAMILY_WISE_PASS
    previous = float(sizing.calibration_pass(study.MARGINS, count - 400, np.array([1.0]), 0.95)[0])
    assert previous**2 < sizing.FAMILY_WISE_PASS
    assert sizing.calibration_control_probability(study.MARGINS, count, 0.85) > 0.999


def test_the_measured_causes_are_pending_until_the_run() -> None:
    # A number of causes joins MEASURED_CAUSES only when every gated row of its cell and every
    # property cell of its law pass in the committed record.
    if not PERFORMANCE.exists():
        assert frozenset({2}) == survival_result.MEASURED_CAUSES
        return
    performance = pd.read_csv(PERFORMANCE)
    cells = pd.read_csv(PROPERTIES)
    expected = {2}
    for cell, law in study.CELLS.items():
        rows = performance.loc[performance["scenario"] == cell]
        prefix = "l1_" if law == "one_cause" else "l3_"
        owned = cells.loc[cells["cell"].str.startswith(prefix)]
        if (
            len(rows) == len(study.ESTIMANDS[cell])
            and rows["passed"].all()
            and owned["passed"].all()
        ):
            expected.add(laws.LAWS[law].causes)
    assert frozenset(expected) == survival_result.MEASURED_CAUSES


def test_a_declared_run_waits_for_the_pilot(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.canonical.continuous_survival_causes import regenerate

    monkeypatch.setattr(sys, "argv", ["regenerate", "--output", "/tmp/x"])
    assert regenerate.pilot_refusals() == []
    monkeypatch.setattr(pilot, "read", pd.DataFrame)
    with pytest.raises(SystemExit, match="run the pilot first"):
        regenerate.refuse_undeclared()
    monkeypatch.setattr(sys, "argv", ["regenerate", "--replicates", "2", "--output", "/tmp/x"])
    regenerate.refuse_undeclared()


# ------------------------------------------------------------ the readings on synthetic rows


def _pilot_rows(
    *, ratio: float = 1.0, failures: int = 0, plug_in_bias: float = 1.0, factual_bias: float = 1.0
) -> pd.DataFrame:
    """Pilot rows like the driver's: calibrated and unbiased, with chosen departures."""
    rng = np.random.default_rng(5)
    records = []
    for cell, law in study.CELLS.items():
        truth = laws.truth(law)
        early = set(laws.first_quartile(law))
        noise = rng.normal(size=(study.PILOT_REPLICATES, len(truth)))
        noise = (noise - noise.mean(axis=0)) / noise.std(axis=0, ddof=1)
        for replicate in range(study.PILOT_REPLICATES):
            head = {"kind": "primary", "label": cell, "replicate": replicate, "n": 1_000}
            if replicate < failures:
                records.append({**head, "refusal": "DataError"})
                continue
            for column, (name, value) in enumerate(truth.items()):
                error = 0.01 * noise[replicate, column]
                records.append(
                    {
                        **head,
                        "estimand": name,
                        "early": name in early,
                        "truth": value,
                        "estimate": value + error,
                        "std_error": 0.01 * ratio,
                        "ic_std_error": 0.01 * ratio,
                        "covered": int(abs(error) <= 1.96 * 0.01 * ratio),
                        "initial_estimate": value + error,
                        "refusal": "",
                        "seconds": 1.0,
                    }
                )
    for law, labels in properties.ROBUSTNESS_LABELS.items():
        truth = laws.truth(law)
        noise = rng.normal(size=(pilot.CONTROL_DRAWS, 3))
        noise = (noise - noise.mean(axis=0)) / noise.std(axis=0, ddof=1)
        for replicate in range(pilot.CONTROL_DRAWS):
            for label, name in labels.items():
                value = truth[name]
                records.append(
                    {
                        "kind": "controls",
                        "label": f"{law}_controls",
                        "replicate": replicate,
                        "n": 1_000,
                        "estimand": name,
                        "robustness_label": label,
                        "truth": value,
                        "estimate": value + 0.01 * noise[replicate, 0],
                        "std_error": 0.01,
                        "initial_estimate": value + 0.01 * (noise[replicate, 1] + plug_in_bias),
                        "factual_estimate": value + 0.01 * (noise[replicate, 2] + factual_bias),
                        "factual_std_error": 0.01,
                        "refusal": "",
                        "seconds": 2.0,
                        "factual_seconds": 2.0,
                    }
                )
    return pd.DataFrame(records)


def test_the_pilot_readings_run_on_rows_like_the_drivers(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.canonical.continuous_survival_causes import regenerate

    rows = _pilot_rows()
    primary = pilot.primary_count(rows)
    calibration = pilot.calibration_count(rows)
    assert primary >= study.MINIMUM_REPLICATES
    # Two calibrated cells pooled in one arm: at 800 pilot draws the predictive SD of the arm's
    # ratio (about 0.018) sets the count well above the at-target count.
    assert calibration == 5_200 > pilot.at_target_count()
    assert (pilot.calibration_checks(rows, calibration)["plug_in_pass"] >= 0.90).all()
    readings = pilot.control_readings(rows)
    assert len(readings) == 9 and readings["informative"].all()
    assert (readings.loc[readings["role"] == "positive", "pass_probability"] > 0.99).all()
    monkeypatch.setattr(pilot, "read", lambda: rows)
    monkeypatch.setattr(study, "PRIMARY_REPLICATES", primary)
    monkeypatch.setattr(properties, "CALIBRATION_REPLICATES", calibration)
    # The synthetic pilot holds the declared 800 draws, not the extension.
    assert "the pilot of l3_cf_1k does not hold its 2000 draws" in regenerate.pilot_refusals()
    monkeypatch.setattr(pilot, "EXTENSION_DRAWS", study.PILOT_REPLICATES)
    assert regenerate.pilot_refusals() == []
    monkeypatch.setattr(study, "PRIMARY_REPLICATES", primary + 400)
    assert regenerate.pilot_refusals() == [
        f"the pilot sets {primary} primary replicates and the record declares {primary + 400}"
    ]
    monkeypatch.setattr(study, "PRIMARY_REPLICATES", primary)
    # Each departure is refused by name: lost fits, a control with no power, a low SE ratio.
    departures = {
        "loses more than": _pilot_rows(failures=20),
        "as uninformative": _pilot_rows(plug_in_bias=0.1),
        "above the cap": _pilot_rows(ratio=0.95),
    }
    for message, frame in departures.items():
        monkeypatch.setattr(pilot, "read", lambda frame=frame: frame)
        reasons = regenerate.pilot_refusals()
        assert any(message in reason for reason in reasons), (message, reasons)


def test_a_cell_whose_pilot_predicts_its_failure_bound_fails_refuses() -> None:
    # Nonzero witness: 1 of 800 passes at 3,200 replicates; 12 of 800 (1.5%) refuses.
    assert pilot.predicted_exceedance(1, 800, 3_200, 32) < 0.01
    assert pilot.predicted_exceedance(12, 800, 3_200, 32) > 0.5
