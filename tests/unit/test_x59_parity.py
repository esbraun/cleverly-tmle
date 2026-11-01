"""X59 section 4: the exact parity gates, their committed tables and the sizing rule.

``parity.csv`` is recomputed from ``replicates.parquet`` and ``parity-replicates.csv`` for every
registered study that commits one (M5), so a hand edit or a stale table fails here.  The gate
arithmetic is checked on synthetic rows with a deliberate mutation, because an exact-parity
check on two equal columns cannot tell a working gate from one that reads nothing.
"""

from __future__ import annotations

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


def test_the_sizing_rule_reproduces_the_plans_worked_examples() -> None:
    """``X59-plan.md`` section 6 at the rev 3.1 rates: 0.91 and 0.99 CPU-h per replicate."""
    timings = {"exact": 0.91, "margins": 0.99}
    tiers = {"exact": 2, "margins": 4}
    twelve = parity.reference_sizes(
        timings, tiers, workers=12, tier1_cpu_hours=30.0, tier3_cpu_hours=17.0
    )
    assert (twelve.tier2, twelve.tier4) == (100, 100)
    assert twelve.cost + 47.0 == pytest.approx(237.0)
    assert twelve.wall_hours + 47.0 / 12 == pytest.approx(19.75)
    eight = parity.reference_sizes(
        timings, tiers, workers=8, tier1_cpu_hours=30.0, tier3_cpu_hours=17.0
    )
    assert (eight.tier2, eight.tier4) == (50, 50)
    assert eight.counts == {"exact": 50, "margins": 50}


def test_a_tier_that_fits_no_prefix_pairs_nothing() -> None:
    sizing = parity.reference_sizes(
        {"exact": 5.0, "margins": 5.0},
        {"exact": 2, "margins": 4},
        workers=4,
        tier1_cpu_hours=30.0,
        tier3_cpu_hours=17.0,
    )
    assert sizing.tier2 == 0 and sizing.counts == {}


def test_the_worker_count_follows_the_memory_rule() -> None:
    assert parity.worker_count(32_000, 2_000) == 12
    assert parity.worker_count(16_000, 2_000) == 6
    assert parity.worker_count(1_000, 2_000) == 1
