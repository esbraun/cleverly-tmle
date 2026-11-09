"""The CI shards between them run every test exactly once."""

from __future__ import annotations

import pytest

from tests import sharding


def test_every_unit_lands_in_exactly_one_shard() -> None:
    units = [f"tests/unit/test_x.py::test_{i}" for i in range(500)] + ["study-a", "study-b"]
    for count in (1, 2, 3, 4):
        seen = [sharding.shard_of(unit, count) for unit in units]
        assert set(seen) <= set(range(1, count + 1))
        if count > 1:
            sizes = [seen.count(i) for i in range(1, count + 1)]
            assert min(sizes) > len(units) / count * 0.7, sizes


def test_the_hook_keeps_groups_whole_and_the_shards_partition_the_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from tests.conftest import _keep_this_shard

    def make(nodeid: str, group: str | None) -> SimpleNamespace:
        marker = None if group is None else SimpleNamespace(args=(group,), kwargs={})
        return SimpleNamespace(nodeid=nodeid, get_closest_marker=lambda _name, m=marker: m)

    items = [make(f"t::{i}", f"g{i % 7}" if i % 2 else None) for i in range(200)]
    seen: list[str] = []
    for index in (1, 2, 3):
        monkeypatch.setenv(sharding.SHARD_ENV, f"{index}/3")
        part = list(items)
        config = SimpleNamespace(hook=SimpleNamespace(pytest_deselected=lambda items: None))
        _keep_this_shard(config, part)  # type: ignore[arg-type]
        seen += [item.nodeid for item in part]
        groups = {i.get_closest_marker("x").args[0] for i in part if i.get_closest_marker("x")}
        for group in groups:
            members = {
                i.nodeid for i in items if (m := i.get_closest_marker("x")) and m.args[0] == group
            }
            assert members <= {i.nodeid for i in part}
    assert sorted(seen) == sorted(item.nodeid for item in items)


@pytest.mark.parametrize("value", ["0/3", "4/3", "x", "1/y", "1"])
def test_a_malformed_shard_is_refused(value: str) -> None:
    with pytest.raises(ValueError, match="CLEVERLY_TEST_SHARD"):
        sharding.parse(value)


def test_an_empty_value_means_no_sharding() -> None:
    assert sharding.parse("") is None
    assert sharding.parse("2/3") == (2, 3)


def test_the_ci_job_shards_each_interpreter_and_keeps_the_old_check_names() -> None:
    from tests.conftest import ROOT

    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    job = workflow.split("\n  tests:", 1)[1].split("\n  # ", 1)[0]
    assert "CLEVERLY_TEST_SHARD: ${{ matrix.shard }}/${{ matrix.shards }}" in job
    assert job.count("shards: 4") == 4 and job.count("shards: 2") == 4
    verdict = workflow.split("\n  tests-verdict:", 1)[1].split("\n  # ", 1)[0]
    assert "name: tests\n" in verdict
    assert "needs: tests" in verdict
