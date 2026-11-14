"""``StudyRecord.point_only_scenarios``: truth rows pass on bias, paired rows skip calibration."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from tests.studies import canonical_adaptive_iptw_pairing as pairing
from tests.studies.evidence import performance


def _group(scenario: str, std_error: float) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    estimate = 0.5 + 0.01 * rng.standard_normal(400)
    return pd.DataFrame(
        {
            "implementation": pairing.IMPLEMENTATION,
            "scenario": scenario,
            "replicate": np.arange(400),
            "estimand": "ey0",
            "truth": 0.5,
            "estimate": estimate,
            "inference_estimate": estimate,
            "inference_scale": "identity",
            "std_error": std_error,
            # An SE half the spread: coverage and the SE ratio fail, the bias leg passes.
            "covered": (np.abs(estimate - 0.5) <= 1.96 * std_error).astype(int),
        }
    )


@pytest.mark.parametrize(("point_only", "expected"), [(True, True), (False, False)])
def test_a_point_only_truth_row_passes_on_its_bias_leg(point_only: bool, expected: bool) -> None:
    scenarios = frozenset({"p_insample"}) if point_only else frozenset()
    record = dataclasses.replace(pairing.STUDY, point_only_scenarios=scenarios)
    key = (pairing.IMPLEMENTATION, "p_insample", "ey0")
    row = performance._cell((record, _group("p_insample", 0.005), key, 1))
    assert row["bias_equivalent"] and not row["se_calibrated"]
    assert row["passed"] is expected
    assert row.get("point_only", False) is point_only


def test_a_point_only_paired_cell_runs_no_calibration_leg() -> None:
    record = pairing.STUDY
    assert not record.se_comparable("p_insample", "ey0")
    assert record.se_comparable("pd_known_arms", "ey0")


def test_point_only_scenarios_must_be_declared() -> None:
    with pytest.raises(ValueError, match="undeclared"):
        dataclasses.replace(pairing.STUDY, point_only_scenarios=frozenset({"no_such"}))


def test_the_pairing_study_declares_every_estimated_mechanism_point_only() -> None:
    assert pairing.STUDY.point_only_scenarios == frozenset(pairing.SCENARIOS) - pairing.KNOWN


@pytest.mark.parametrize("reported", [True, False])
def test_a_reported_truth_row_passes_by_declaration(reported: bool) -> None:
    scenarios = frozenset({"p_insample"}) if reported else frozenset()
    record = dataclasses.replace(
        pairing.STUDY, reported_scenarios=scenarios, point_only_scenarios=frozenset()
    )
    key = (pairing.IMPLEMENTATION, "p_insample", "ey0")
    row = performance._cell((record, _group("p_insample", 0.005), key, 1))
    # The legs are published either way; only the declaration changes the verdict.
    assert not row["se_calibrated"]
    assert row["passed"] is reported
    assert row.get("reported", False) is reported
    assert record.se_comparable("p_insample", "ey0") is (not reported)


def test_reported_scenarios_must_be_declared() -> None:
    with pytest.raises(ValueError, match="undeclared"):
        dataclasses.replace(pairing.STUDY, reported_scenarios=frozenset({"no_such"}))


@pytest.mark.parametrize(
    ("scenario", "lost", "gate_passed", "reason"),
    [
        ("v3_half_n", 72, True, "bound exceeded, reported cell"),
        ("v3_half_n", 10, True, "within bound"),
        ("v1_rare_cause2", 72, False, "bound exceeded"),
        ("v1_rare_cause2", 10, True, "within bound"),
    ],
)
def test_a_reported_cell_publishes_its_failure_bound_and_gates_nothing(
    scenario: str, lost: int, gate_passed: bool, reason: str
) -> None:
    """A reported cell over its bound passes the gate with its red leg published; a gated
    cell over the same bound still fails (``continuous-survival-variance``: V3 reported, V1
    gated, 0.01 of 6,000 replicates allows 60)."""
    from tests.studies import continuous_survival_variance as variance
    from tests.studies.evidence import failures

    record = variance.STUDY
    assert failures.allowed(record, record.implementation, scenario) == 60
    rows = failures._rows(
        table="performance",
        implementation=record.implementation,
        scenario=scenario,
        estimand=variance.ESTIMANDS[scenario][0],
        record=record,
        failed=lost,
        dropped=lost,
        legs=[],
        counted={record.implementation: lost},
    )
    (row,) = rows
    assert row["leg"] == "failure bound"
    assert row["leg_passed"] is (lost <= 60)
    assert row["gate_passed"] is gate_passed
    assert row["reason"] == reason
    assert failures.red(pd.DataFrame(rows)).empty is gate_passed
