"""X59 section 4: the exact parity gates, their committed tables and the sizing rule.

``parity.csv`` is recomputed from ``replicates.parquet`` and ``parity-replicates.csv`` for every
registered study that commits one (M5), so a hand edit or a stale table fails here.  The gate
arithmetic is checked on synthetic rows with a deliberate mutation, because an exact-parity
check on two equal columns cannot tell a working gate from one that reads nothing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.studies.evidence import parity
from tests.studies.evidence.manifest import read_rows
from tests.studies.evidence.registry import registered
from tests.unit import _x59_reference_only_support as synthetic

STUDIES = [study for study in registered() if study.artifact("parity.csv").exists()]


@pytest.mark.parametrize("study", STUDIES, ids=[study.slug for study in STUDIES])
def test_the_committed_parity_table_is_recomputed_from_the_rows(study) -> None:  # type: ignore[no-untyped-def]
    declaration = study.runner().PARITY
    rows = read_rows(study.artifact("replicates.parquet"))
    inputs = pd.read_csv(study.artifact("parity-replicates.csv"), float_precision="round_trip")
    recomputed = parity.parity_table(rows, inputs, record=study, declaration=declaration)
    committed = pd.read_csv(study.artifact("parity.csv"), float_precision="round_trip")
    pd.testing.assert_frame_equal(recomputed, committed, check_dtype=False, rtol=1e-12)


def _rows(shift: float = 0.0, se_shift: float = 0.0) -> pd.DataFrame:
    record = synthetic.compared()
    rng = np.random.default_rng(5)
    out = []
    for scenario in ("a", "b"):
        for replicate in range(record.replicates_of(synthetic.REFERENCE, scenario)):
            estimate = float(rng.normal())
            mutated = scenario == "a" and replicate == 3
            for implementation, delta, se_delta in (
                (synthetic.SUBJECT, 0.0, 0.0),
                (synthetic.REFERENCE, shift * mutated, se_shift * mutated),
            ):
                out.append(
                    {
                        "implementation": implementation,
                        "scenario": scenario,
                        "replicate": replicate,
                        "estimand": f"ey[{scenario}]",
                        "estimate": estimate + delta,
                        "std_error": 0.1 + se_delta,
                    }
                )
    return pd.DataFrame(out)


def _inputs(rows: pd.DataFrame, move: float, gcomp_shift: float = 0.0) -> pd.DataFrame:
    subject = rows.loc[rows["implementation"] == synthetic.SUBJECT]
    plugin = subject["estimate"].to_numpy() - move
    return pd.DataFrame(
        {
            "scenario": subject["scenario"].to_numpy(),
            "replicate": subject["replicate"].to_numpy(),
            "estimand": subject["estimand"].to_numpy(),
            "subject_plugin": plugin,
            "reference_gcomp": plugin + gcomp_shift,
        }
    )


def _table(rows: pd.DataFrame, inputs: pd.DataFrame) -> pd.DataFrame:
    return parity.parity_table(
        rows, inputs, record=synthetic.compared(), declaration=synthetic.PARITY
    )


def test_equal_columns_pass_every_gate_and_the_witness() -> None:
    rows = _rows()
    table = _table(rows, _inputs(rows, move=0.01))
    assert table["passed"].all()
    assert parity.failed_gates(table).empty
    assert parity.witness_scenarios(table) == {"a": True}
    counts = table.set_index(["gate", "quantity"])["paired_replicates"]
    assert counts[("G-P", "estimate")] == 10 and counts[("G1", "estimate")] == 8


@pytest.mark.parametrize(
    ("shift", "se_shift", "gcomp_shift", "gate"),
    [(2e-6, 0.0, 0.0, "G-P"), (0.0, 2e-6, 0.0, "G-P"), (0.0, 0.0, 2e-6, "G-C")],
)
def test_one_replicate_over_its_tolerance_fails_its_gate(
    shift: float, se_shift: float, gcomp_shift: float, gate: str
) -> None:
    """A nonzero witness: the mutated column moves one gate, and the table names the replicate."""
    rows = _rows(shift=shift, se_shift=se_shift)
    table = _table(rows, _inputs(rows, move=0.01, gcomp_shift=gcomp_shift))
    failed = parity.failed_gates(table)
    assert set(failed["gate"]) == {gate}, failed
    if shift or se_shift:
        assert set(failed["max_replicate"]) == {3}
    assert parity.mutation_detected(table, synthetic.PARITY) is (
        max(shift, se_shift, gcomp_shift) >= parity.MUTATION_FACTOR * 1e-6
    )


def test_a_large_mutation_is_detected_at_ten_tolerances() -> None:
    rows = _rows(shift=1e-4)
    table = _table(rows, _inputs(rows, move=0.01))
    assert parity.mutation_detected(table, synthetic.PARITY)


def test_an_update_that_barely_moves_carries_no_targeting_claim() -> None:
    rows = _rows()
    table = _table(rows, _inputs(rows, move=1e-5))
    assert parity.witness_scenarios(table) == {"a": False}
    assert parity.failed_gates(table).empty, "a witness row is a claim's reach, not a gate"


def test_an_undeclared_tolerance_refuses_to_read() -> None:
    declaration = parity.ParityDeclaration(cells=(parity.ParityCell("G-P", "a", "ey[a]"),))
    rows = _rows()
    with pytest.raises(ValueError, match="not declared yet"):
        parity.parity_table(rows, None, record=synthetic.compared(), declaration=declaration)


def test_the_declared_tolerance_is_ten_smoke_maxima_above_the_floor() -> None:
    assert parity.declared_tolerance(1e-9) == parity.PARITY_FLOOR
    assert parity.declared_tolerance(3e-7) == pytest.approx(3e-6)
    rows = _rows(shift=4e-7)
    maxima = parity.smoke_maxima(_table(rows, _inputs(rows, move=0.01)))
    assert maxima["G-P"] == pytest.approx(4e-7)


def test_a_streamed_run_fits_each_read_but_its_carried_replicate_then_the_tail() -> None:
    # The smoke's eight replicates in one read of thirteen: seven fitted, then the tail.
    assert parity.stream_rounds(8, 3, 13) == 3 + 1
    # Fifty in reads of twelve on twelve workers: 11, 12, 12, 12, 2, then the tail.
    assert parity.stream_rounds(50, 12, 12) == 5 + 1
    assert parity.stream_rounds(1, 4, 4) == 1
    assert parity.stream_rounds(0, 4, 4) == 0


def test_a_read_above_the_worker_count_costs_a_second_round() -> None:
    """The deliberate mutation: a chunk of ``w + 1`` on ``w`` workers doubles the rounds."""
    # 100 in reads of twelve: 11, then seven of 12, then 4 and the tail.
    assert parity.stream_rounds(100, 12, 12) == 1 + 7 + 1 + 1
    # In reads of thirteen: 12, then six of 13 in two rounds each, then 9 and the tail.
    assert parity.stream_rounds(100, 12, 13) == 1 + 12 + 1 + 1


def _cost(seconds: float, workers: int, tier: int, replicates: int | None = None) -> Any:
    return parity.ScenarioCost(
        seconds=seconds, workers=workers, chunk=workers, tier=tier, replicates=replicates
    )


def test_the_sizing_rule_charges_each_scenario_on_its_own_workers() -> None:
    costs = {
        "light": _cost(10.0, 12, 2),
        "heavy": _cost(600.0, 3, 2),
        "margins": _cost(60.0, 6, 4),
        "parity": _cost(5.0, 12, parity.PARITY_TIER, replicates=500),
    }
    sizing = parity.reference_sizes(costs, tier1_hours=0.0)
    three = 1 + parity.stream_rounds(500, 12, 12)
    assert sizing.hours[parity.PARITY_TIER] == pytest.approx(5.0 * three / 3600.0)
    left = parity.RUN_HOURS - sizing.hours[parity.PARITY_TIER]
    assert sizing.hours[2] <= parity.TIER2_SHARE * left
    bigger = [c for c in parity.TIER2_COUNTS if c > sizing.tier2]
    if bigger:
        assert parity.tier_hours(costs, 2, bigger[0]) > parity.TIER2_SHARE * left
    assert sizing.hours[4] <= left - sizing.hours[2]
    assert sizing.counts["parity"] == 500
    assert sizing.counts["heavy"] == sizing.tier2 and sizing.counts["margins"] == sizing.tier4
    assert sizing.wall_hours == pytest.approx(sum(sizing.hours.values()))
    # The same heavy scenario on one worker instead of three takes a smaller tier-2 prefix.
    starved = parity.reference_sizes({**costs, "heavy": _cost(600.0, 1, 2)}, tier1_hours=0.0)
    assert starved.tier2 < sizing.tier2


def test_contention_and_tier1_take_wall_hours_from_the_tiers() -> None:
    costs = {"exact": _cost(900.0, 4, 2), "margins": _cost(300.0, 4, 4)}
    free = parity.reference_sizes(costs, tier1_hours=0.0)
    slowed = parity.reference_sizes(costs, tier1_hours=6.0, contention=1.3)
    assert free.tier2 == 200 and slowed.tier2 == 100


def test_a_tier_that_fits_no_prefix_pairs_nothing() -> None:
    sizing = parity.reference_sizes(
        {"exact": _cost(3600.0, 1, 2), "margins": _cost(3600.0, 1, 4)}, tier1_hours=0.0
    )
    assert sizing.tier2 == 0 and sizing.counts == {}


def test_only_the_parity_tier_fixes_its_replicates() -> None:
    with pytest.raises(ValueError, match="fixes its replicates"):
        parity.reference_sizes({"exact": _cost(1.0, 1, 2, replicates=50)}, tier1_hours=0.0)
    with pytest.raises(ValueError, match="fixes its replicates"):
        parity.reference_sizes({"parity": _cost(1.0, 1, parity.PARITY_TIER)}, tier1_hours=0.0)
