"""``available_cpus``: the affinity, capped by the smallest quota among the process's cgroups."""

from __future__ import annotations

from pathlib import Path

import pytest

from cleverly.utils import parallel


def _tree(tmp_path: Path, cgroup: str, files: dict[str, str]) -> tuple[str, str]:
    root = tmp_path / "fs"
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (tmp_path / "cgroup").write_text(cgroup)
    return str(tmp_path / "cgroup"), str(root)


def test_a_nested_v2_quota_is_read_from_the_process_s_own_group(tmp_path: Path) -> None:
    cgroup, root = _tree(
        tmp_path,
        "0::/system.slice/job.service\n",
        {
            "cpu.max": "max 100000\n",
            "system.slice/cpu.max": "max 100000\n",
            "system.slice/job.service/cpu.max": "150000 100000\n",
        },
    )
    assert parallel._cgroup_cpus(cgroup, root) == 1.5


def test_the_smallest_ancestor_quota_wins(tmp_path: Path) -> None:
    cgroup, root = _tree(
        tmp_path,
        "0::/a/b\n",
        {"a/cpu.max": "200000 100000\n", "a/b/cpu.max": "max 100000\n"},
    )
    assert parallel._cgroup_cpus(cgroup, root) == 2.0


def test_no_quota_reads_none(tmp_path: Path) -> None:
    cgroup, root = _tree(tmp_path, "0::/\n", {"cpu.max": "max 100000\n"})
    assert parallel._cgroup_cpus(cgroup, root) is None


def test_a_v1_quota_is_read(tmp_path: Path) -> None:
    cgroup, root = _tree(
        tmp_path,
        "4:cpu,cpuacct:/docker/abc\n3:memory:/docker/abc\n",
        {
            "cpu/docker/abc/cpu.cfs_quota_us": "300000\n",
            "cpu/docker/abc/cpu.cfs_period_us": "100000\n",
            "cpu/cpu.cfs_quota_us": "-1\n",
            "cpu/cpu.cfs_period_us": "100000\n",
        },
    )
    assert parallel._cgroup_cpus(cgroup, root) == 3.0


def test_a_container_s_own_group_is_read_at_the_mount_root(tmp_path: Path) -> None:
    cgroup, root = _tree(tmp_path, "0::/not/mounted/here\n", {"cpu.max": "50000 100000\n"})
    assert parallel._cgroup_cpus(cgroup, root) == 0.5


def test_malformed_files_read_no_quota(tmp_path: Path) -> None:
    cgroup, root = _tree(tmp_path, "0::/\n", {"cpu.max": "lots\n"})
    assert parallel._cgroup_cpus(cgroup, root) is None


def test_the_quota_caps_the_affinity_and_rounds_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cgroup, root = _tree(tmp_path, "0::/\n", {"cpu.max": "150000 100000\n"})
    monkeypatch.setattr(parallel, "CGROUP", cgroup)
    monkeypatch.setattr(parallel, "CGROUP_ROOT", root)
    monkeypatch.setattr(parallel.os, "sched_getaffinity", lambda pid: set(range(8)), raising=False)
    assert parallel.available_cpus() == 2
    monkeypatch.setattr(parallel.os, "sched_getaffinity", lambda pid: {0}, raising=False)
    assert parallel.available_cpus() == 1
