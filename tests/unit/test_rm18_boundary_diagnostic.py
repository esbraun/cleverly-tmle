"""The RM18 BD diagnostic and FW-B: pass probabilities, gate rules, the paired rules, the harness."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.diagnostics import rm18_shared as shared
from tests.diagnostics.rm18_boundary import paired
from tests.diagnostics.rm18_boundary import run as bd
from tests.studies.canonical_multi_arm_drtmle import STUDY as MULTI
from tests.studies.canonical_weighted_ltmle_crossfit import STUDY as CROSSFIT
from tests.studies.evidence.seeds import stream_seed

#: BD-0 as the declaration publishes it.
DECLARED_BD0 = (0.849, 0.998, 0.981, 0.849, 0.834)


def test_bd0_reproduces_the_declared_pass_probabilities() -> None:
    table = shared.reading_frame(bd.bd0_table())
    assert tuple(round(float(value), 3) for value in table["value"]) == DECLARED_BD0


def _summary(**columns: Any) -> Any:
    defaults = {
        "property": "root_n_and_efficiency",
        "passed": False,
        "coverage_ci_lower": 0.93,
        "coverage_ci_upper": 0.96,
        "bias_discriminated": False,
        "bias_equivalent": True,
        "se_ratio": 1.0,
        "se_ratio_ci_lower": 0.95,
        "se_ratio_ci_upper": 1.05,
    }
    return next(pd.DataFrame([{**defaults, **columns}]).itertuples(index=False))


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (_summary(passed=True), bd.SATISFIES),
        (_summary(), bd.UNRESOLVED),
        (_summary(coverage_ci_upper=0.899), bd.FAILS_GATE),
        (_summary(bias_discriminated=True), bd.FAILS_GATE),
        (_summary(se_ratio=1.21), bd.FAILS_GATE),
        (_summary(property=bd.CONTRACTION_FAMILY, coverage_ci_upper=0.899), bd.FAILS_GATE),
        (_summary(property=bd.CONTRACTION_FAMILY, coverage_ci_upper=0.901), bd.UNRESOLVED),
        (_summary(property="interval_calibration", se_ratio_ci_upper=0.929), bd.FAILS_GATE),
        (_summary(property="interval_calibration", coverage_ci_lower=0.981), bd.FAILS_GATE),
        (_summary(property="interval_calibration", se_ratio_ci_upper=1.08), bd.UNRESOLVED),
        (_summary(property="double_robustness"), bd.FAILS_GATE),
        (_summary(property="double_robustness", bias_equivalent=False), bd.UNRESOLVED),
        (
            _summary(property="double_robustness", bias_equivalent=False, se_ratio=11.0),
            bd.FAILS_GATE,
        ),
    ],
)
def test_the_gate_reading_on_each_kind_and_its_mutations(row: Any, expected: str) -> None:
    assert bd.gate_label(row, MULTI.margins) == expected


def _rung(covered: int) -> pd.DataFrame:
    count = 6_000
    rng = np.random.default_rng(4)
    return pd.DataFrame(
        {
            "property": bd.CONTRACTION_FAMILY,
            "cell": "outcome_correct_n4000",
            "role": "positive",
            "replicate": np.arange(count),
            "n": 4_000,
            "requested_replicates": count,
            "failed_replicates": 0,
            "truth": 0.1,
            "estimate": 0.1 + 0.01 * rng.normal(size=count),
            "std_error": 0.01,
            "covered": (np.arange(count) < covered).astype(int),
            "rejected": 0,
        }
    )


def test_the_registered_rung_rule_reads_the_fresh_rows() -> None:
    def label(covered: int) -> str:
        summary = bd.registered_summary(_rung(covered), MULTI, "outcome_correct_n4000")
        return bd.gate_label(next(summary.itertuples(index=False)), MULTI.margins)

    assert label(5_700) == bd.SATISFIES
    assert label(5_280) == bd.FAILS_GATE
    assert label(5_450) == bd.UNRESOLVED


@pytest.mark.parametrize(
    ("resolution", "expected"),
    [
        (0.1121, 16_086),
        (0.2, paired.BUDGET_CAP),
        (0.02, paired.BUDGET_FLOOR),
        (float("nan"), paired.BUDGET_CAP),
    ],
)
def test_the_step_2_budget(resolution: float, expected: int) -> None:
    pilot = pd.DataFrame(
        [
            {
                "convention": convention,
                "estimand": estimand,
                "calibration_excess_resolution": resolution / (1 + index),
            }
            for convention in paired.CONVENTIONS
            for index, estimand in enumerate(paired.MEANS)
        ]
    )
    assert paired.paired_budget(pilot)[1] == expected
    with pytest.raises(RuntimeError, match="every mean row"):
        paired.paired_budget(pilot.iloc[1:])


@pytest.fixture(scope="module")
def committed_comparisons() -> pd.DataFrame:
    return paired.comparisons(paired.committed_paired(), CROSSFIT, 1)


def test_step_1_native_reproduces_the_committed_equivalence(
    committed_comparisons: pd.DataFrame,
) -> None:
    held, largest, compared = paired.reproduces_committed(committed_comparisons)
    assert held and compared == 3, largest


def test_the_hajek_convention_moves_every_mean_row(committed_comparisons: pd.DataFrame) -> None:
    """The nonzero witness: the substitution reaches the comparison."""
    by = committed_comparisons.set_index(["convention", "estimand"])
    for estimand in paired.MEANS:
        assert (
            by.loc[("hajek", estimand), "calibration_excess_upper"]
            != by.loc[("native", estimand), "calibration_excess_upper"]
        )
    stored = shared.read_rows(CROSSFIT.artifact("reference-inference.csv.gz"))
    hajek = paired.with_convention(paired.committed_paired(), "hajek")
    lmtp = hajek.loc[hajek["implementation"] == paired.COMPARATOR].merge(
        stored.loc[stored["inference_method"] == "hajek"],
        on=["replicate", "estimand"],
        suffixes=("", "_stored"),
    )
    assert len(lmtp) == 3 * CROSSFIT.replicates
    assert (lmtp["covered"] == lmtp["covered_stored"]).all()


def _row(committed: pd.DataFrame, estimand: str, **changes: Any) -> Any:
    base = (
        committed.loc[(committed["estimand"] == estimand) & (committed["convention"] == "native")]
        .iloc[0]
        .to_dict()
    )
    return next(pd.DataFrame([{**base, **changes}]).itertuples(index=False))


PASS = {
    "paired_ci_lower": -0.001,
    "paired_ci_upper": 0.001,
    "rmse_ratio_upper": 1.0,
    "coverage_difference_lower": 0.0,
    "calibration_excess_upper": 0.01,
    "calibration_excess_resolution": 0.01,
}


def test_the_bd_p_reading_and_its_mutations(committed_comparisons: pd.DataFrame) -> None:
    estimand = paired.BD_ROWS[0]
    passing = _row(committed_comparisons, estimand, **PASS)
    failing = _row(committed_comparisons, estimand, **{**PASS, "coverage_difference_lower": -0.03})
    assert paired.bd_p_label(passing, failing) == paired.EQUIVALENT
    assert paired.bd_p_label(failing, passing) == paired.CONVENTION
    assert paired.bd_p_label(failing, failing) == "native inconclusive; hajek inconclusive"


def test_the_fw_b_reading_and_its_mutations(committed_comparisons: pd.DataFrame) -> None:
    row = paired.FW_ROW
    passing = _row(committed_comparisons, row, **PASS)
    coverage_fails = {**PASS, "coverage_difference_lower": -0.03}
    native = _row(committed_comparisons, row, **coverage_fails)
    assert paired.fw_b_label(passing, native) == paired.EQUIVALENT
    assert paired.fw_b_label(native, passing) == paired.CONVENTION
    low = {**coverage_fails, "subject_coverage_ci_upper": 0.915}
    assert (
        paired.fw_b_label(
            _row(committed_comparisons, row, **low), _row(committed_comparisons, row, **low)
        )
        == paired.DEFICIT
    )
    # Each clause of the deficit reading is load bearing.
    covered = {**low, "subject_coverage_ci_upper": 0.921}
    assert (
        paired.fw_b_label(
            _row(committed_comparisons, row, **covered), _row(committed_comparisons, row, **covered)
        )
        == paired.UNRESOLVED
    )
    wide = {**low, "calibration_excess_resolution": 0.051}
    assert (
        paired.fw_b_label(
            _row(committed_comparisons, row, **low), _row(committed_comparisons, row, **wide)
        )
        == paired.UNRESOLVED
    )


def test_fresh_seeds_miss_every_registered_seed() -> None:
    for part in ("BD-1", "BD-2"):
        for record, module, property_name, cell, declared in bd.CELLS[part]:
            labels = ("rm18", bd.DESIGN, property_name, cell)
            if module is bd.binary:
                order = [(*labels, index) for index in range(declared)]
                seeds = shared.fresh_seeds(record, order, bd.binary_registered_seeds())
                assert len(set(seeds)) == declared
                assert set(seeds).isdisjoint(bd.binary_registered_seeds())
                continue
            registered = bd.coverage_registered_states(record, module)
            seed = bd.coverage_seed(record, labels, declared, registered)
            states = np.random.SeedSequence(seed).generate_state(declared).tolist()
            assert len(set(states)) == declared and registered.isdisjoint(states)
    control = [payload[5] for payload in bd.control_payloads(bd.CONTROL_REPLICATES)]
    assert len(set(control)) == len(control)
    assert set(control).isdisjoint(shared.weighted_registered_seeds(CROSSFIT))
    assert paired.disjoint()


def test_the_registered_set_holds_the_sl_budget() -> None:
    """Without SL's 73,000 outer-rung seeds the calibration re-read would keep its label."""
    labels = ("rm18", bd.DESIGN, "interval_calibration", "correctly_specified")
    registered = bd.coverage_registered_states(MULTI, bd.multi)
    assert bd.coverage_seed(MULTI, labels, bd.CALIBRATION_REPLICATES, registered) != stream_seed(
        MULTI, *labels
    )


@pytest.mark.parametrize(
    "entry",
    [entry for part in ("BD-1", "BD-2") for entry in bd.CELLS[part]],
    ids=lambda entry: f"{entry[0].slug}:{entry[3]}",
)
def test_two_registered_replicates_of_each_cell_reproduce(entry: tuple[Any, ...]) -> None:
    record, module, property_name, cell, _ = entry
    rows = pd.DataFrame(bd.validate_cell(record, module, property_name, cell, "BD", 2, 1))
    assert rows["result"].eq(shared.HOLDS).all(), rows.to_string()


def test_one_registered_control_replicate_reproduces() -> None:
    validation = shared.validate_weighted(
        CROSSFIT,
        cross_fit=True,
        part="BD-3",
        payload=("double_robustness", "both_wrong"),
        cells=[bd.CONTROL],
        cap=1,
        jobs=1,
    )
    assert validation["result"].eq(shared.HOLDS).all(), validation.to_string()


def test_b_inf_of_the_control_is_finite() -> None:
    assert math.isfinite(bd.control_limit())


@pytest.mark.parametrize("part", [part for part in bd.PARTS if part != "BD-P2"])
def test_each_committed_reading_follows_from_its_rows(part: str) -> None:
    _, _, reading = shared.part_paths(bd.HERE, part)
    if not reading.exists():
        pytest.skip(f"{part} has not run")
    rebuilt = bd.reading_table(part, bd.HERE, 1)
    pd.testing.assert_frame_equal(rebuilt, shared.read_rows(reading), check_dtype=False, rtol=1e-12)


def test_the_committed_step_2_readings_follow_from_its_comparisons() -> None:
    _, _, reading = shared.part_paths(bd.HERE, "BD-P2")
    if not reading.exists():
        pytest.skip("BD-P2 has not run")
    rebuilt = bd.reading_table("BD-P2", bd.HERE, 1)
    pd.testing.assert_frame_equal(rebuilt, shared.read_rows(reading), check_dtype=False, rtol=1e-12)
    pilot = shared.read_rows(bd.HERE / "pilot.csv")
    rows = shared.read_rows(shared.part_paths(bd.HERE, "BD-P2")[0])
    assert rows["replicate"].nunique() == paired.paired_budget(pilot)[1]
