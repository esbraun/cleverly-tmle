"""A study whose scenarios run at different sample sizes, and a study whose scenarios do not.

``StudyRecord.scenario_n`` gives a scenario its own size.  A record that leaves it empty is a
single-size study, and every reader must treat it exactly as before the field existed: the same
declaration hash, the same resume key, the same manifest configuration, the same row check.
A record that sets it carries the sizes into each of those, and an override of ``--n`` runs every
scenario at the override.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from tests.canonical import regenerate
from tests.studies import canonical_longitudinal_sdr
from tests.studies.evidence import claims
from tests.studies.evidence.manifest import read_rows, write_manifest
from tests.studies.evidence.schema import validate_replicates

SINGLE = dataclasses.replace(
    canonical_longitudinal_sdr.STUDY,
    reference=None,
    comparison=None,
    comparison_overrides={},
    failure_bound={
        side: bound
        for side, bound in canonical_longitudinal_sdr.STUDY.failure_bound.items()
        if side == canonical_longitudinal_sdr.STUDY.implementation
    },
)
SCENARIOS = tuple(SINGLE.scenarios)


def _sized(**changes: Any) -> Any:
    sizes = {SCENARIOS[0]: SINGLE.n * 2}
    return dataclasses.replace(SINGLE, **{"scenario_n": sizes, **changes})


def _rows(record: Any) -> pd.DataFrame:
    """The committed subject rows, with ``n`` set to each scenario's declared size."""
    committed = read_rows(canonical_longitudinal_sdr.STUDY.artifact("replicates.parquet"))
    rows = committed.loc[committed["implementation"] == record.implementation].copy()
    rows["n"] = rows["scenario"].map(record.sizes(record.n))
    return rows.reset_index(drop=True)


def test_the_study_under_test_has_several_scenarios() -> None:
    assert len(SCENARIOS) >= 2


class TestASingleSizeRecord:
    def test_every_scenario_runs_at_n(self) -> None:
        assert SINGLE.scenario_n == {}
        assert SINGLE.sizes(SINGLE.n) == dict.fromkeys(SCENARIOS, SINGLE.n)
        assert {SINGLE.n_of(name) for name in SCENARIOS} == {SINGLE.n}

    def test_the_declaration_hash_ignores_the_empty_field(self) -> None:
        declared = regenerate.canonical(SINGLE)
        declared.pop("artifacts")
        assert declared.pop("scenario_n") == {}
        assert regenerate.declaration_hash(SINGLE) == regenerate.digest(declared)

    def test_the_resume_key_has_no_sizes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(regenerate, "git_state", lambda: {})
        assert "scenario_n" not in regenerate.run_key(SINGLE)

    def test_the_manifest_records_no_sizes(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        write_manifest(path, dataclasses.replace(SINGLE, artifacts=tmp_path), [], configuration={})
        configuration = json.loads(path.read_text(encoding="utf-8"))["configuration"]
        assert "scenario_n" not in configuration
        assert configuration["n"] == SINGLE.n

    def test_the_row_check_is_unchanged(self) -> None:
        rows = _rows(SINGLE)
        validate_replicates(rows, record=SINGLE)
        rows.loc[0, "n"] = SINGLE.n + 1
        with pytest.raises(ValueError, match=f"sample size other than {SINGLE.n}$"):
            validate_replicates(rows, record=SINGLE)


class TestAPerScenarioRecord:
    def test_a_named_scenario_takes_its_size_and_the_rest_take_n(self) -> None:
        record = _sized()
        assert record.n_of(SCENARIOS[0]) == 2 * SINGLE.n
        assert all(record.n_of(name) == SINGLE.n for name in SCENARIOS[1:])

    def test_an_override_runs_every_scenario_at_the_override(self) -> None:
        assert _sized().sizes(200) == dict.fromkeys(SCENARIOS, 200)

    @pytest.mark.parametrize(
        ("sizes", "match"),
        [
            ({"no such scenario": 100}, "unknown scenarios"),
            ({SCENARIOS[0]: 0}, "positive integers"),
            ({SCENARIOS[0]: 1.5}, "positive integers"),
            ({SCENARIOS[0]: True}, "positive integers"),
        ],
    )
    def test_a_wrong_size_is_refused(self, sizes: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            dataclasses.replace(SINGLE, scenario_n=sizes)

    def test_a_seed_owner_and_its_follower_share_one_size(self) -> None:
        with pytest.raises(ValueError, match="owner's size"):
            _sized(scenario_seed_owners={SCENARIOS[1]: SCENARIOS[0]})
        same = {SCENARIOS[0]: 500, SCENARIOS[1]: 500}
        dataclasses.replace(
            SINGLE, scenario_n=same, scenario_seed_owners={SCENARIOS[1]: SCENARIOS[0]}
        )

    def test_the_sizes_enter_the_declaration_and_the_resume_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(regenerate, "git_state", lambda: {})
        record = _sized()
        assert regenerate.declaration_hash(record) != regenerate.declaration_hash(SINGLE)
        other = _sized(scenario_n={SCENARIOS[0]: 3 * SINGLE.n})
        assert regenerate.declaration_hash(record) != regenerate.declaration_hash(other)
        assert regenerate.run_key(record)["scenario_n"] == {SCENARIOS[0]: 2 * SINGLE.n}

    def test_the_manifest_records_the_sizes(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        write_manifest(path, _sized(artifacts=tmp_path), [], configuration={})
        configuration = json.loads(path.read_text(encoding="utf-8"))["configuration"]
        assert configuration["scenario_n"] == {SCENARIOS[0]: 2 * SINGLE.n}

    def test_the_row_check_reads_each_scenarios_size(self) -> None:
        record = _sized()
        rows = _rows(record)
        validate_replicates(rows, record=record)
        # A row of the sized scenario at the base n is refused, as a base row at the sized n is.
        for scenario, wrong in ((SCENARIOS[0], SINGLE.n), (SCENARIOS[1], 2 * SINGLE.n)):
            broken = rows.copy()
            broken.loc[broken.index[broken["scenario"] == scenario][0], "n"] = wrong
            with pytest.raises(ValueError, match="sample size other than"):
                validate_replicates(broken, record=record)

    def test_one_bound_needs_one_size(self) -> None:
        estimand = SINGLE.scenarios[SCENARIOS[0]][0]
        shared = [name for name in SCENARIOS if estimand in SINGLE.scenarios[name]]
        record = _sized(efficiency_bounds={estimand: 1.0})
        if len(shared) > 1:
            with pytest.raises(ValueError, match="cannot be published"):
                claims.thresholds(record)
        sized = dataclasses.replace(record, scenario_n=dict.fromkeys(shared, 400))
        assert claims.thresholds(sized)[f"bound:{estimand}_standard_error"] == pytest.approx(
            1.0 / 400**0.5
        )


def test_the_driver_records_the_sizes_and_drops_them_on_an_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``main`` keys the run by the declared sizes, and by none when ``--n`` overrides them."""
    record = _sized(artifacts=tmp_path)
    seen: list[Any] = []

    def stop(scratch: Path, key: Any, *, fresh: bool) -> None:
        seen.append(key)
        raise SystemExit(0)

    monkeypatch.setattr(regenerate, "git_state", lambda: {})
    monkeypatch.setattr(regenerate, "runs_root", lambda: tmp_path / "runs")
    monkeypatch.setattr(regenerate, "_claim_scratch", stop)
    for n in (record.n, 200):
        arguments = argparse.Namespace(replicates=record.replicates, n=n, cache=None)
        monkeypatch.setattr(regenerate, "_arguments", lambda *args, a=arguments: a)
        study = SimpleNamespace(STUDY=record)
        with pytest.raises(SystemExit):
            regenerate.main(study, SimpleNamespace(), here=tmp_path)
    assert seen[0]["scenario_n"] == {SCENARIOS[0]: 2 * SINGLE.n}
    assert "scenario_n" not in seen[1]
    assert seen[1]["n"] == 200
