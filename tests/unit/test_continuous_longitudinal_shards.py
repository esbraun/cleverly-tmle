"""The sharded declared run of ``continuous-time-longitudinal``: its shards and their merge.

The fits are replaced by a cheap stand-in that is a function of the key alone, as a real fit
is through its seed, so the test runs the shard, merge and publication code at a smoke count
(``--replicates``) in the fast tier.  The stand-in writes no wall time; a real run's
``seconds`` columns are the one reading a rerun does not reproduce.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.canonical.continuous_time_longitudinal import regenerate
from tests.studies import continuous_time_longitudinal as study
from tests.studies import continuous_time_longitudinal_law as laws
from tests.studies.evidence.seeds import replicate_seed

REPLICATES = 12
#: The key whose stand-in fit raises a recordable error, so the merge joins a failure record.
FAILED = ("cont_cf", 1)


def _draw(key: tuple[str, int, int]) -> None:
    del key


def _replicate(key: tuple[str, int, int], drawn: None) -> list[dict[str, Any]]:
    """A calibrated stand-in for one fit: normal estimates around the truth, seeded by key."""
    del drawn
    scenario, replicate, n = key
    if (scenario, replicate) == FAILED:
        raise np.linalg.LinAlgError("a singular stand-in design")
    rng = np.random.default_rng(replicate_seed(study.STUDY, scenario, replicate))
    truth = laws.TRUTHS[study.law_of(scenario)]
    rows = []
    for name in study.ESTIMANDS[scenario]:
        se = 0.02
        estimate = truth[name] + rng.normal(0.0, se)
        low, high = estimate - 1.96 * se, estimate + 1.96 * se
        reference = 4.0 if name.startswith("ate_regimen") else np.nan
        caps = {
            "cap": 4.0,
            "reference_cap": reference,
            "capped_truth": laws.capped_truth(study.law_of(scenario), name, 4.0, reference),
        }
        row: dict[str, Any] = {
            "implementation": study.SUBJECT,
            "scenario": scenario,
            "replicate": replicate,
            "n": n,
            "estimand": name,
            "truth": truth[name],
            "estimate": estimate,
            "inference_estimate": estimate,
            "std_error": se,
            "ci_lower": low,
            "ci_upper": high,
            "inference_scale": "identity",
            "interval_scale": "identity",
            "covered": int(low <= truth[name] <= high),
            "initial_estimate": estimate + 0.01,
            "seconds": 1.0,
            "steps": 10.0,
            "converged": 1.0,
            "bounded_share": 0.0,
            "jump_bound_share": 0.0,
            "visit_bound_share": 0.0,
            "grid_size": 100.0,
            "dropped_visits": 0.0,
            "largest_cap": 4.0,
            "variance_rel_sd": 0.05,
            "event_share": 0.1,
            "followers_at_last": 100.0,
            "past_cap_share": 0.0,
            **caps,
        }
        for cell, (_, count, names) in study.SECONDARY.items():
            ran = scenario == "day_cf" and name in names and replicate < count
            paired = cell == "ltmle_daily"
            centre = laws.PANEL_TRUTHS.get(name, np.nan) if paired else truth[name]
            value = centre + rng.normal(0.0, se)
            secondary_caps = dict.fromkeys(caps, np.nan) if paired else caps
            row.update(
                {
                    f"{cell}_estimate": value if ran else np.nan,
                    f"{cell}_std_error": se if ran else np.nan,
                    f"{cell}_covered": float(abs(value - centre) <= 1.96 * se) if ran else np.nan,
                    f"{cell}_initial": value + 0.01 if ran else np.nan,
                    f"{cell}_failed": 0.0 if ran else np.nan,
                    f"{cell}_interval_refused": 0.0 if ran else np.nan,
                    f"{cell}_seconds": 1.0 if ran else np.nan,
                    **{
                        f"{cell}_{field}": figure if ran else np.nan
                        for field, figure in secondary_caps.items()
                    },
                    f"{cell}_error": "",
                }
            )
        rows.append(row)
    return rows


@pytest.fixture
def stand_in(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Stand-in fits, and the shared driver's scratch under ``tmp_path``."""
    monkeypatch.setattr(study, "_replicate", _replicate)
    monkeypatch.setattr(study, "_draw", _draw)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    return tmp_path


def _smoke(output: Path, *flags: str) -> None:
    regenerate.main(
        [
            "--replicates",
            str(REPLICATES),
            "--output",
            str(output),
            "--jobs",
            "2",
            "--allow-failures",
            *flags,
        ]
    )


def _published(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir())}


def test_the_shards_cover_the_run_once() -> None:
    keys = study.shard_keys(study.PRIMARY_REPLICATES)
    shards = [study.shard_keys(study.PRIMARY_REPLICATES, (i, 3)) for i in (1, 2, 3)]
    assert sorted(key for shard in shards for key in shard) == sorted(keys)
    sizes = [len(shard) for shard in shards]
    assert max(sizes) - min(sizes) <= len(study.STUDY.scenarios)
    # Every shard holds a third of the X70 pairing, the costliest fit after cont_cf's.
    paired = [
        sum(cell == "day_cf" and replicate < study.PAIRED_REPLICATES for cell, replicate in shard)
        for shard in shards
    ]
    assert max(paired) - min(paired) <= 1
    with pytest.raises(ValueError, match="shard 4/3"):
        study.shard_keys(10, (4, 3))


def test_a_merged_run_is_byte_identical_to_an_unsharded_one(
    stand_in: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    single, merged = stand_in / "single", stand_in / "merged"
    shards = [stand_in / "shard-1", stand_in / "shard-2"]
    _smoke(single)
    for index, shard in enumerate(shards, start=1):
        _smoke(shard, "--shard", f"{index}/2")
    record = json.loads((shards[1] / regenerate.SHARD_FILE).read_text(encoding="utf-8"))
    assert record["complete"] and record["keys"] == REPLICATES and record["shard"] == "2/2"
    assert (shards[1] / regenerate.SHARD_FAILURES).exists()  # cont_cf replicate 1 raised
    assert not (shards[0] / regenerate.SHARD_FAILURES).exists()
    capsys.readouterr()
    _smoke(merged, "--merge", *map(str, shards))
    # The merge fits nothing: every replicate comes from a shard's checkpoint.
    assert f"reusing {2 * REPLICATES} checkpointed replicates" in capsys.readouterr().out

    one, joined = _published(single), _published(merged)
    assert set(one) == set(joined)
    assert "failures.csv" in one
    differing = [name for name in one if name != "manifest.json" and one[name] != joined[name]]
    assert differing == []
    single_manifest = json.loads(one["manifest.json"])
    merged_manifest = json.loads(joined["manifest.json"])
    recorded = merged_manifest["generated_with"].pop("shards")
    assert merged_manifest == single_manifest
    assert [entry["shard"] for entry in recorded] == ["1/2", "2/2"]
    assert {entry["commit"] for entry in recorded} == {
        single_manifest["generated_with"]["subject"]["cleverly_commit"]
    }
    assert all(entry["seconds"] >= 0 for entry in recorded)


def test_the_merge_refuses_a_gap_an_overlap_and_an_incomplete_shard(stand_in: Path) -> None:
    first, second = stand_in / "shard-1", stand_in / "shard-2"
    _smoke(first, "--shard", "1/2")
    _smoke(second, "--shard", "2/2")
    with pytest.raises(SystemExit, match="gap: shard \\[2\\] of 2"):
        _smoke(stand_in / "merged", "--merge", str(first))
    with pytest.raises(SystemExit, match="overlap: shard \\[1\\] of 2"):
        _smoke(stand_in / "merged", "--merge", str(first), str(first), str(second))
    stamp = second / regenerate.SHARD_FILE
    record = json.loads(stamp.read_text(encoding="utf-8"))
    missing = next((second / regenerate.SHARD_CHECKPOINTS).rglob("*.pkl.gz"))
    missing.unlink()
    with pytest.raises(SystemExit, match="misses the checkpoint"):
        _smoke(stand_in / "merged", "--merge", str(first), str(second))
    stamp.write_text(json.dumps({**record, "complete": False}), encoding="utf-8")
    with pytest.raises(SystemExit, match=r"shard 2/2 in .* is incomplete"):
        _smoke(stand_in / "merged", "--merge", str(first), str(second))
    stamp.write_text(json.dumps({**record, "head": "0" * 40}), encoding="utf-8")
    with pytest.raises(SystemExit, match="different commits"):
        _smoke(stand_in / "merged", "--merge", str(first), str(second))


def test_the_merge_refuses_smoke_shards_for_a_declared_run(stand_in: Path) -> None:
    """Review SH3: a smoke shard never joins a declared record."""
    shards = [stand_in / "shard-1", stand_in / "shard-2"]
    for index, shard in enumerate(shards, start=1):
        _smoke(shard, "--shard", f"{index}/2")
    key = regenerate.run_key(REPLICATES, study.PRIMARY_N)
    assert regenerate.merge_refusals(shards, key, REPLICATES, declared=False) == []
    reasons = regenerate.merge_refusals(shards, key, REPLICATES, declared=True)
    assert [reason for reason in reasons if "is a smoke shard" in reason] == [
        f"shard {index}/2 in {shard} is a smoke shard" for index, shard in enumerate(shards, 1)
    ]


def test_the_merge_warns_when_the_shards_ran_on_different_machines(
    stand_in: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Review SH1: the resume key holds no CPU model or BLAS kernel, so shard.json does."""
    shards = [stand_in / "shard-1", stand_in / "shard-2"]
    for index, shard in enumerate(shards, start=1):
        _smoke(shard, "--shard", f"{index}/2")
    stamp = shards[1] / regenerate.SHARD_FILE
    record = json.loads(stamp.read_text(encoding="utf-8"))
    assert record["machine"] == regenerate.machine() and record["machine"]["cpu"]
    capsys.readouterr()
    _smoke(stand_in / "same", "--merge", *map(str, shards))
    assert "different machines" not in capsys.readouterr().err
    record["machine"] = {"cpu": "another CPU", "blas": ["openblas 0.0 Haswell"]}
    stamp.write_text(json.dumps(record), encoding="utf-8")
    _smoke(stand_in / "mixed", "--merge", *map(str, shards))
    assert "warning: the shards ran on different machines" in capsys.readouterr().err
    machines = json.loads((stand_in / "mixed" / "manifest.json").read_text(encoding="utf-8"))
    assert machines["generated_with"]["shards"][1]["machine"]["cpu"] == "another CPU"


def test_a_declared_run_takes_its_five_flags_only() -> None:
    assert regenerate.FLAGS == ("--output", "--jobs", "--fresh", "--shard", "--merge")
    message = "takes --output, --jobs, --fresh, --shard, --merge only"
    with pytest.raises(SystemExit, match=message):
        regenerate.main(["--output", "/tmp/elsewhere", "--cache", "/tmp/cache"])
    with pytest.raises(SystemExit, match="a shard or a merge, not both"):
        regenerate.main(["--shard", "1/3", "--merge", "a", "b"])
    for text in ("0/3", "4/3", "1/1", "one/3"):
        with pytest.raises(SystemExit, match="--shard"):
            regenerate.parse_shard(text)
    assert regenerate.parse_shard("2/3") == (2, 3)
