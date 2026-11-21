"""The committed run of ``refuter-sampled`` (X106 H), recomputed from its trials.

``trials.csv`` holds every trial's readings and printed verdicts. These tests rebuild each gated
verdict in ``properties.csv`` from it, check that it agrees with the replicate table, and check
the declared verdict scope: every trial on the registered configuration (``I2``) prints a verdict
for each rule, and every trial below it (``I1``) or on clusters prints none. The study adds no
verdict-registry row: a sampled call reads the registry at its sampled baseline's configuration.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pandas as pd
import pytest

from tests.studies import refuter_sampled as study
from tests.studies import refuter_sampled_properties as properties

ARTIFACTS = study.STUDY.artifacts


@cache
def _trials() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / study.TRIALS_FILE, keep_default_na=False)


@cache
def _properties() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / "properties.csv").set_index(["property", "cell"])


def _rejections(name: str, reading: str, *, trials: int | None = None) -> np.ndarray:
    own = _trials().loc[_trials()["cell"] == name].sort_values("trial")
    values = own[f"{reading}_rejected"]
    if trials is not None:
        values = values.iloc[:trials]
    return values.astype(int).to_numpy()


def _passed(family: str, cell: str) -> bool:
    return bool(_properties().loc[(family, cell), "passed"])


def test_every_trial_was_recorded_once() -> None:
    counts = _trials().groupby("cell")["trial"].agg(["count", "nunique"])
    for name, cell in study.CELLS.items():
        assert counts.loc[name, "count"] == counts.loc[name, "nunique"] == cell.trials


def test_the_trials_agree_with_the_replicate_table() -> None:
    replicates = pd.read_parquet(ARTIFACTS / "property-replicates.parquet")
    for family, cells in study.PROPERTY_CELLS.items():
        for cell in cells:
            name, reading = cell.split("__", 1)
            rows = replicates.loc[(replicates["property"] == family) & (replicates["cell"] == cell)]
            rows = rows.sort_values("replicate")
            expected = _rejections(name, reading, trials=len(rows))
            assert np.array_equal(rows["rejected"].astype(int).to_numpy(), expected), cell


@pytest.mark.parametrize("cell", study.PROPERTY_CELLS[study.SIZE_FAMILY])
def test_each_size_verdict_recomputes(cell: str) -> None:
    name, reading = cell.split("__")
    rejected = _rejections(name, reading)
    assert rejected.size == study.SIZE_TRIALS
    assert _passed(study.SIZE_FAMILY, cell) is properties.size_passes(
        int(rejected.sum()), rejected.size
    )
    assert int(rejected.sum()) <= study.SIZE_MAX_REJECTIONS


@pytest.mark.parametrize("cell", study.PROPERTY_CELLS[study.POWER_FAMILY])
def test_each_power_verdict_recomputes(cell: str) -> None:
    name, defect = cell.split("__")
    detected = _rejections(name, defect, trials=study.POWER_TRIALS)
    assert _passed(study.POWER_FAMILY, cell) is (int(detected.sum()) >= study.POWER_MIN_DETECTIONS)


@pytest.mark.parametrize("cell", study.PROPERTY_CELLS[study.RELATIVE_FAMILY])
def test_each_relative_verdict_recomputes(cell: str) -> None:
    name, reading = cell.split("__")
    own = _rejections(name, reading)
    partner = _rejections(study.CELLS[name].partner, reading)
    assert _passed(study.RELATIVE_FAMILY, cell) is properties.relative_passes(own, partner)


@pytest.mark.parametrize("cell", study.PROPERTY_CELLS[study.MUTATION_FAMILY])
def test_each_mutation_is_detected(cell: str) -> None:
    # The mutation reads the full fit's SE and unit count in place of the sampled baseline's.
    # Both of its legs must fail the gates the sampled reading passes.
    name = cell.split("__")[0]
    mutated = _rejections(name, "subset_full_reference")
    partner = _rejections(study.CELLS[name].partner, "subset")
    assert not properties.size_passes(int(mutated.sum()), mutated.size)
    assert properties.relative_detects(mutated, partner)
    assert _passed(study.MUTATION_FAMILY, cell)


def test_a_sampled_reading_passes_the_gate_the_mutation_fails() -> None:
    # The gate functions are not trivially red: on the same cells the sampled subset reading
    # passes both the absolute and the relative gate.
    for name in study.MUTATION_CELLS:
        sampled = _rejections(name, "subset")
        partner = _rejections(study.CELLS[name].partner, "subset")
        assert properties.size_passes(int(sampled.sum()), sampled.size)
        assert properties.relative_passes(sampled, partner)


def _shares() -> list[dict[str, object]]:
    records = []
    for name, group in _trials().groupby("cell", sort=False):
        record: dict[str, object] = {"cell": name}
        for reading in study.READINGS:
            record[f"{reading}_verdict_share"] = float((group[f"{reading}_verdict"] != "").mean())
        records.append(record)
    return records


def test_the_printed_verdicts_keep_the_declared_scope() -> None:
    assert all(properties.scope_problems(record) == [] for record in _shares())


def test_a_clustered_verdict_would_break_the_scope() -> None:
    # Deliberate-mutation control on the scope check.
    record = next(record for record in _shares() if record["cell"] == "C-T")
    assert properties.scope_problems({**record, "placebo_verdict_share": 0.5}) == [
        "placebo printed a verdict on 0.500 of trials"
    ]


@pytest.mark.parametrize("name", study.GATED_CELLS)
def test_each_printed_verdict_matches_its_rejection(name: str) -> None:
    own = _trials().loc[_trials()["cell"] == name]
    for reading in study.READINGS:
        verdict = own[f"{reading}_verdict"].map({"True": True, "False": False})
        assert (verdict == (own[f"{reading}_rejected"].astype(int) == 0)).all(), reading


def test_a_binding_limit_sampled_and_a_slack_one_did_not() -> None:
    for name, cell in study.CELLS.items():
        applied = set(_trials().loc[_trials()["cell"] == name, "sample_applied"].astype(str))
        binding = cell.m is not None and cell.m < cell.n
        assert applied == {str(binding)}, name
