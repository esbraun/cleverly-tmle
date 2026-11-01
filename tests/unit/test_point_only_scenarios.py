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
