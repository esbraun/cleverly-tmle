"""The RM18 BD diagnostic and FW-B: pass probabilities, gate rules, the paired rules, the seeds."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.diagnostics import rm18_seeds
from tests.diagnostics import rm18_shared as shared
from tests.diagnostics.rm18_boundary import paired
from tests.diagnostics.rm18_boundary import run as bd
from tests.studies import drtmle_properties as binary
from tests.studies.canonical_drtmle import STUDY as BINARY
from tests.studies.canonical_multi_arm_drtmle import STUDY as MULTI
from tests.studies.canonical_weighted_ltmle_crossfit import STUDY as CROSSFIT
from tests.studies.evidence.property_verdicts import apply_shared_verdicts, contraction_verdicts
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


def _rung(covered: int, count: int = 6_000) -> pd.DataFrame:
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
            "seed": np.arange(count),
            "study": MULTI.slug,
        }
    )


def test_the_registered_rung_rule_reads_the_fresh_rows_and_a_short_rung_is_smoke() -> None:
    def label(rows: pd.DataFrame) -> str:
        readings = bd.cell_readings(
            "BD-1", bd.registered_summary(rows, MULTI, "rung"), MULTI, 6_000
        )
        return str(next(row["result"] for row in readings if row["statistic"] == "reading"))

    assert label(_rung(5_700)) == bd.SATISFIES
    # The rung rule reads coverage alone.  A bias interval wholly outside 0.25 SD leaves the
    # reading where the coverage interval puts it, as the declared rung row states.
    biased = _rung(5_700).assign(estimate=lambda rows: rows["estimate"] + 0.005)
    assert bd.registered_summary(biased, MULTI, "rung")["bias_discriminated"].iloc[0]
    assert label(biased) == bd.SATISFIES
    assert label(_rung(5_280)) == bd.FAILS_GATE
    assert label(_rung(5_450)) == bd.UNRESOLVED
    assert label(_rung(5_700, count=5_999)) == shared.SMOKE


@pytest.mark.parametrize(
    ("resolution", "expected"),
    [(0.1121, 16_086), (0.2, paired.BUDGET_CAP), (0.02, paired.BUDGET_FLOOR)],
)
def test_the_step_2_budget(resolution: float, expected: int) -> None:
    assert paired.paired_budget(_pilot(resolution), smoke=False)[1] == expected
    with pytest.raises(RuntimeError, match="every mean row"):
        paired.paired_budget(_pilot(resolution).iloc[1:], smoke=False)


def _pilot(resolution: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "convention": convention,
                "estimand": estimand,
                "calibration_excess_upper": 0.0,
                "calibration_excess_resolution": resolution / (1 + index),
            }
            for convention in paired.CONVENTIONS
            for index, estimand in enumerate(paired.MEANS)
        ]
    )


def test_a_non_finite_r_pilot_stops_a_declared_step_2() -> None:
    pilot = _pilot(0.05)
    pilot.loc[3, "calibration_excess_resolution"] = np.nan
    with pytest.raises(RuntimeError, match="non-finite r_pilot"):
        paired.paired_budget(pilot, smoke=False)
    resolution, budget = paired.paired_budget(pilot, smoke=True)
    assert math.isnan(resolution) and budget == paired.BUDGET_CAP


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


PASS = {
    "paired_ci_lower": -0.001,
    "paired_ci_upper": 0.001,
    "rmse_ratio_upper": 1.0,
    "coverage_difference_lower": 0.0,
    "calibration_excess_upper": 0.01,
    "calibration_excess_resolution": 0.01,
    "subject_coverage_ci_upper": 0.97,
}
COVERAGE_FAILS = {**PASS, "coverage_difference_lower": -0.03}
DEFICIT = {**COVERAGE_FAILS, "subject_coverage_ci_upper": 0.915}


def _compared(committed: pd.DataFrame, legs: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """The committed comparisons with the legs of each ``convention`` replaced, on every row."""
    frame = committed.copy()
    for convention, values in legs.items():
        mask = frame["convention"] == convention
        for column, value in values.items():
            frame.loc[mask, column] = value
    return frame


def _labels(compared: pd.DataFrame, smoke: bool = False) -> dict[tuple[str, str], str]:
    rows = paired.comparison_readings("BD-P2", compared, labelled=True, smoke=smoke)
    return {
        (row["part"], row["scope"]): str(row["result"])
        for row in rows
        if row["statistic"] == "reading"
    }


def test_the_paired_readings_through_the_table(committed_comparisons: pd.DataFrame) -> None:
    fw_b = ("FW-B", paired.FW_ROW)
    bd_p = ("BD-P", paired.BD_ROWS[0])
    labels = _labels(_compared(committed_comparisons, {"native": PASS, "hajek": COVERAGE_FAILS}))
    assert labels[fw_b] == labels[bd_p] == paired.EQUIVALENT
    labels = _labels(_compared(committed_comparisons, {"native": COVERAGE_FAILS, "hajek": PASS}))
    assert labels[fw_b] == paired.CONVENTION and labels[bd_p] == paired.CONVENTION
    labels = _labels(_compared(committed_comparisons, {"native": DEFICIT, "hajek": DEFICIT}))
    assert labels[fw_b] == paired.DEFICIT
    assert labels[bd_p] == "native inconclusive, hajek inconclusive"
    # Each clause of the deficit reading is load bearing.
    covered = {**DEFICIT, "subject_coverage_ci_upper": 0.921}
    assert (
        _labels(_compared(committed_comparisons, {"native": covered, "hajek": covered}))[fw_b]
        == paired.UNRESOLVED
    )
    wide = {**DEFICIT, "calibration_excess_resolution": 0.051}
    assert (
        _labels(_compared(committed_comparisons, {"native": DEFICIT, "hajek": wide}))[fw_b]
        == paired.UNRESOLVED
    )
    assert set(_labels(committed_comparisons, smoke=True).values()) == {shared.SMOKE}


def test_the_step_2_table_is_rebuilt_from_its_comparisons(
    committed_comparisons: pd.DataFrame, tmp_path: Path
) -> None:
    rows_path, validation_path, _ = shared.part_paths(tmp_path, "BD-P2")
    shared.write_table(_pilot(0.02), tmp_path / "pilot.csv")
    shared.write_table(pd.DataFrame({"replicate": range(paired.BUDGET_FLOOR)}), rows_path)
    shared.write_table(
        shared.validation_frame([shared.validation_row("BD-P2", "the pilot", (True, 0.0, 1))]),
        validation_path,
    )
    compared = _compared(committed_comparisons, {"native": COVERAGE_FAILS, "hajek": PASS})
    shared.write_table(compared, paired.comparisons_path(tmp_path, "BD-P2"))
    table = bd.table("BD-P2", tmp_path, 1)
    readings = table.loc[table["statistic"] == "reading"].set_index("scope")["result"]
    assert readings[paired.FW_ROW] == paired.CONVENTION
    # A step 2 short of R_p is a smoke run.
    shared.write_table(pd.DataFrame({"replicate": range(4)}), rows_path)
    table = bd.table("BD-P2", tmp_path, 1)
    assert set(table.loc[table["statistic"] == "reading", "result"]) == {shared.SMOKE}


# ---------------------------------------------------------------------------------- seeds


def test_every_part_draws_disjoint_seeds_on_each_record() -> None:
    for record, parts in rm18_seeds.every_part().items():
        names = list(parts)
        for part in names:
            assert len(set(parts[part])) == len(parts[part]), (record, part)
        for first in range(len(names)):
            for second in range(first + 1, len(names)):
                overlap = set(parts[names[first]]) & set(parts[names[second]])
                assert not overlap, (record, names[first], names[second], sorted(overlap)[:3])


def test_fresh_seeds_miss_every_registered_seed() -> None:
    parts = rm18_seeds.every_part()
    crossfit = shared.weighted_registered_seeds(CROSSFIT)
    for name, seeds in parts[CROSSFIT.slug].items():
        assert crossfit.isdisjoint(seeds), name
    assert rm18_seeds.binary_registered().isdisjoint(parts[rm18_seeds.BINARY.slug]["BD-1"])
    for slug, (record, module) in rm18_seeds.COVERAGE_RECORDS.items():
        registered = rm18_seeds.coverage_registered(record, module)
        for name, states in parts[slug].items():
            assert registered.isdisjoint(states), (slug, name)


def test_the_registered_set_holds_the_sl_budget() -> None:
    """Without SL's 73,000 outer-rung seeds the calibration re-read would keep its label."""
    labels = ("rm18", bd.DESIGN, "interval_calibration", "correctly_specified")
    seed = rm18_seeds.coverage_seeds()[MULTI.slug, "interval_calibration", "correctly_specified"]
    assert seed != stream_seed(MULTI, *labels)


def test_the_bd_3_payloads_carry_their_declared_seeds() -> None:
    seeds = rm18_seeds.weighted_seeds()["BD-3"]
    assert [payload[2][5] for payload in bd.control_payloads()] == list(seeds)
    assert [payload[2][5] for payload in bd.control_payloads(2)] == list(seeds[:2])


# --------------------------------------------------------------------------------------- R4


@pytest.fixture(scope="module")
def validations() -> dict[str, pd.DataFrame]:
    """Two registered replicates of every re-read cell, one summary per study, once."""
    return {part: bd.validate_cells(part, 2, 1) for part in bd.CELLS}


@pytest.mark.parametrize("part", ["BD-1", "BD-2"])
def test_two_registered_replicates_of_each_cell_reproduce(
    validations: dict[str, pd.DataFrame], part: str
) -> None:
    validation = validations[part]
    assert validation["result"].eq(shared.HOLDS).all(), validation.to_string()
    assert len(validation) == len(bd.CELLS[part]) + len({entry[0].slug for entry in bd.CELLS[part]})


def test_one_registered_control_replicate_reproduces() -> None:
    validation = shared.validation_frame(
        shared.validate_weighted(
            CROSSFIT,
            cross_fit=True,
            part="BD-3",
            payload=("double_robustness", "both_wrong"),
            cells=[bd.CONTROL],
            cap=1,
            jobs=1,
        )
    )
    assert validation["result"].eq(shared.HOLDS).all(), validation.to_string()


def test_b_inf_of_the_control_is_finite() -> None:
    assert math.isfinite(bd.control_limit())


# ------------------------------------------------------------------------ the committed run


DECLARED_ROWS = {
    "BD-1": lambda rows: rows.groupby(["study", "property", "cell"]).size().eq(6_000).all(),
    "BD-2": lambda rows: len(rows) == 17_000,
    "BD-3": lambda rows: rows.groupby("cell").size().eq(bd.CONTROL_REPLICATES).all(),
    "BD-P1": lambda rows: rows["replicate"].nunique() == CROSSFIT.replicates,
    "BD-P-pilot": lambda rows: rows["replicate"].nunique() == paired.PILOT_REPLICATES,
    "BD-P2": lambda rows: (
        rows["replicate"].nunique()
        == paired.paired_budget(shared.read_rows(bd.HERE / "pilot.csv"), smoke=False)[1]
    ),
}


def _registered_cells() -> list[tuple[Any, str, str]]:
    cells = [
        (entry[0], entry[2], entry[3]) for part in ("BD-1", "BD-2") for entry in bd.CELLS[part]
    ]
    return [*cells, (CROSSFIT, *bd.CONTROL)]


@pytest.mark.parametrize(
    ("record", "property_name", "cell"),
    _registered_cells(),
    ids=lambda value: getattr(value, "slug", value),
)
def test_the_harness_rule_reproduces_each_registered_red_verdict(
    record: Any, property_name: str, cell: str
) -> None:
    """The rule a re-read applies is the registered rule: on the committed registered rows it
    gives the published verdict and coverage interval of the cell."""
    committed = shared.read_rows(record.artifact("property-replicates.csv.gz"))
    rows = committed.loc[(committed["property"] == property_name) & (committed["cell"] == cell)]
    if record.slug == BINARY.slug and property_name == bd.CONTRACTION_FAMILY:
        budget = binary.CONTRACTION_VERDICT_REPLICATES
        rows = rows.loc[rows["replicate"] < budget].assign(requested_replicates=budget)
    summary, _ = apply_shared_verdicts(rows, record, rate_labels=())
    contraction_verdicts(summary, record)
    published = shared.read_rows(record.artifact("properties.csv"))
    expected = published.loc[
        (published["property"] == property_name) & (published["cell"] == cell)
    ].iloc[0]
    row = summary.iloc[0]
    assert not bool(expected["passed"]) and not bool(row["passed"])
    for column in ("coverage_ci_lower", "coverage_ci_upper"):
        assert float(row[column]) == pytest.approx(float(expected[column]), rel=1e-12)
    assert bd.gate_label(row, record.margins) == bd.UNRESOLVED


def test_every_part_of_the_design_has_run() -> None:
    assert set(bd.PARTS) <= shared.RAN


@pytest.mark.parametrize("part", sorted(set(bd.PARTS) & shared.RAN))
def test_each_committed_reading_follows_from_its_record(part: str) -> None:
    rows_path, _, reading = shared.part_paths(bd.HERE, part)
    if part in DECLARED_ROWS:
        assert DECLARED_ROWS[part](shared.read_rows(rows_path))
    rebuilt = bd.table(part, bd.HERE, 1)
    pd.testing.assert_frame_equal(
        shared.as_committed(rebuilt), shared.read_rows(reading), check_dtype=False, rtol=1e-12
    )
    assert shared.SMOKE not in set(rebuilt["result"])
