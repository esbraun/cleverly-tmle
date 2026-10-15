"""The column comparison of the relabel check and the A/B sides of ``replay_committed``."""

from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.canonical import replay_committed
from tests.canonical.replay_committed import equal_entries
from tests.studies.evidence.registry import registered


def test_a_last_bit_difference_passes_against_the_column_scale() -> None:
    left = pd.Series([1.0, 1e-3, 0.0])
    right = pd.Series([1.0 + 2e-16, 1e-3 + 2e-16, 1e-16])
    assert equal_entries(left, right, rtol=1e-12).all()
    assert not equal_entries(left, right, rtol=0.0).all()


def test_an_infinite_entry_does_not_widen_the_tolerance() -> None:
    """The mutation: a scale read with ``nanmax`` is infinite, and every row then passes."""
    left = pd.Series([np.inf, 1.0, 2.0])
    right = pd.Series([np.inf, 1.5, 2.0])
    assert equal_entries(left, right, rtol=1e-12).tolist() == [True, False, True]
    assert not equal_entries(pd.Series([np.inf]), pd.Series([-np.inf]), rtol=1e-12).any()


def test_missing_text_agrees_and_other_text_must_match() -> None:
    left = pd.Series(["a", None, "b"], dtype=object)
    right = pd.Series(["a", np.nan, "c"], dtype=object)
    assert equal_entries(left, right).tolist() == [True, True, False]


# ---------------------------------------------------------------------------------------
# The A/B side: ``replay --write``, ``compare`` and ``replay --all`` (X22 G1).
# ---------------------------------------------------------------------------------------

SOURCE = {
    "tree": "/work/base",
    "head": "a" * 40,
    "diff_sha256": "d" * 64,
    "untracked_src_sha256": "u" * 64,
}


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scenario": ["s", "s", "s", "s"],
            "replicate": [0, 0, 1, 1],
            "estimand": ["ate", "ey1", "ate", "ey1"],
            "estimate": [0.25, 0.5, 0.2500000000001, 0.75],
            "std_error": [0.05, 0.06, 0.05, 0.07],
        }
    )


def _side(
    directory: Path, rows: pd.DataFrame | None = None, **source: str | dict[str, str]
) -> Path:
    """A directory in the layout of ``replay --write`` with the given source fields."""
    (directory / "rows").mkdir(parents=True)
    (_rows() if rows is None else rows).to_csv(directory / "rows" / "demo.csv", index=False)
    meta = {**SOURCE, **source, "errors": {}}
    (directory / "meta.json").write_text(json.dumps(meta))
    return directory


def test_equal_sides_from_different_trees_compare_equal(tmp_path: Path) -> None:
    left = _side(tmp_path / "a")
    right = _side(tmp_path / "b", tree="/work/branch")
    status, report = replay_committed.compare(left, right)
    assert status == 0, report


def test_a_mutated_frame_fails_and_rtol_admits_a_small_one(tmp_path: Path) -> None:
    left = _side(tmp_path / "a")
    moved = _rows()
    moved.loc[1, "estimate"] += 1e-9
    right = _side(tmp_path / "b", moved, tree="/work/branch")
    assert replay_committed.compare(left, right)[0] == 1
    assert replay_committed.compare(left, right, rtol=1e-6)[0] == 0
    # The mutation control: the tolerance is relative to the column's largest magnitude, so a
    # difference larger than that scale fails whatever the tolerance.
    moved.loc[1, "estimate"] += 0.1
    far = _side(tmp_path / "c", moved, tree="/work/branch")
    assert replay_committed.compare(left, far, rtol=1e-6)[0] == 1


def test_a_missing_row_or_study_fails(tmp_path: Path) -> None:
    left = _side(tmp_path / "a")
    right = _side(tmp_path / "b", _rows().iloc[:3], tree="/work/branch")
    assert replay_committed.compare(left, right)[0] == 1
    other = _side(tmp_path / "c", tree="/work/branch")
    (other / "rows" / "demo.csv").rename(other / "rows" / "other.csv")
    assert replay_committed.compare(left, other)[0] == 1


def test_a_recorded_error_fails_the_comparison(tmp_path: Path) -> None:
    left = _side(tmp_path / "a")
    right = _side(tmp_path / "b", tree="/work/branch")
    meta = json.loads((right / "meta.json").read_text())
    meta["errors"] = {"demo": "ValueError: boom"}
    (right / "meta.json").write_text(json.dumps(meta))
    status, report = replay_committed.compare(left, right)
    assert status == 1 and any("boom" in line for line in report)


def test_two_directories_from_one_source_exit_2_and_a_declared_a_a_run_exits_0(
    tmp_path: Path,
) -> None:
    left = _side(tmp_path / "a")
    right = _side(tmp_path / "b")
    assert replay_committed.main(["compare", str(left), str(right)]) == 2
    assert replay_committed.main(["compare", str(left), str(right), "--same-source"]) == 0
    assert replay_committed.compare(left, right)[0] == 2


@pytest.mark.parametrize("field", ["tree", "head", "diff_sha256", "untracked_src_sha256"], ids=str)
def test_any_differing_source_field_compares_normally(tmp_path: Path, field: str) -> None:
    """One tree and commit with a different diff hash is a real B side: compare it."""
    left = _side(tmp_path / "a")
    right = _side(tmp_path / "b", **{field: "changed"})
    assert replay_committed.compare(left, right)[0] == 0


def test_all_names_the_registry_minus_the_exclusions() -> None:
    excluded = {slug for slug, _ in replay_committed.REPLAY_EXCLUDED}
    registry = [record for record in registered() if record.slug not in excluded]
    assert replay_committed.replayable() == tuple(record.slug for record in registry)
    assert excluded <= {record.slug for record in registered()}
    assert all(reason.strip() for _, reason in replay_committed.REPLAY_EXCLUDED)
    # An exclusion is the only way to leave a study out, so a runner without a replay hook
    # must be declared and cannot drop out silently.
    for record in registry:
        module = importlib.import_module(record.runner_module)
        assert hasattr(module, "draw_and_fit"), record.slug


def test_all_replays_every_replayable_study(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []

    def fake(slugs: Any, *args: Any, **kwargs: Any) -> float:
        seen.append((tuple(slugs), args, kwargs))
        return 0.0

    monkeypatch.setattr(replay_committed, "replay", fake)
    assert replay_committed.main(["replay", "--all", "--replicates", "20", "--jobs", "4"]) == 0
    slugs, args, _ = seen[0]
    assert slugs == replay_committed.replayable()
    assert args[:2] == (20, 4)
    with pytest.raises(SystemExit):
        replay_committed.main(["replay"])
    with pytest.raises(SystemExit):
        replay_committed.main(["replay", "--all", "--studies", "weighted-tmle"])


@pytest.fixture(scope="module")
def written(tmp_path_factory: pytest.TempPathFactory) -> Path:
    side = tmp_path_factory.mktemp("replay") / "side"
    worst = replay_committed.replay(
        ["weighted-tmle"], 2, 1, replay_committed.POLICY_LABELS, rtol=1e-9, write=side
    )
    assert worst == 0.0  # within 1e-9 of the committed rows, whatever the platform
    return side


def test_write_stores_the_rows_and_the_source(written: Path) -> None:
    rows = pd.read_csv(written / "rows" / "weighted-tmle.csv", float_precision="round_trip")
    assert list(rows.columns) == [*replay_committed.ROW_KEYS, *replay_committed.REPLAYED]
    assert set(rows["replicate"]) == {0, 1} and len(rows) > 0
    meta = json.loads((written / "meta.json").read_text())
    root = Path(replay_committed.__file__).resolve().parents[2]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert meta["head"] == head and meta["tree"] == str(root)
    assert Path(meta["cleverly_file"]).is_relative_to(root / "src")
    assert meta["replicates"] == 2 and meta["jobs"] == 1 and meta["errors"] == {}
    assert set(meta["threads"]) == set(replay_committed.THREAD_VARIABLES)
    assert len(meta["diff_sha256"]) == 64 and len(meta["untracked_src_sha256"]) == 64


def test_a_written_side_equals_itself_only_as_a_declared_a_a_run(
    written: Path, tmp_path: Path
) -> None:
    again = tmp_path / "again"
    replay_committed.replay(["weighted-tmle"], 2, 1, replay_committed.POLICY_LABELS, write=again)
    assert replay_committed.main(["compare", str(written), str(again)]) == 2
    assert replay_committed.main(["compare", str(written), str(again), "--same-source"]) == 0
