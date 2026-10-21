"""Linux reads the reclaimable memory, not the free pages, and caps it by the cgroup limit.

A CI runner refused a 0.8 GiB density design with 0.6 GiB "available" late in the fast suite.
The free-page count it read leaves out the page cache, which the kernel gives back on demand.

A container sees the memory of the host in ``/proc/meminfo``. The cloud sandbox, for example,
puts its shell in a cgroup v1 group with a 14.3 GB limit, and the out-of-memory killer acts on
that limit and not on ``MemAvailable``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from cleverly.utils import memory

MEMINFO = """MemTotal:       16384000 kB
MemFree:          614400 kB
MemAvailable:    9437184 kB
Buffers:          102400 kB
Cached:          8388608 kB
"""


def test_meminfo_available_reads_the_reclaimable_line(tmp_path: Path) -> None:
    path = tmp_path / "meminfo"
    path.write_text(MEMINFO, encoding="ascii")
    assert memory._meminfo_available(str(path)) == 9437184 * 1024


@pytest.mark.parametrize(
    "text",
    [
        "MemTotal: 16384000 kB\nMemFree: 614400 kB\n",
        "MemAvailable: 9437184 MB\n",
        "MemAvailable:\n",
    ],
    ids=["missing", "unit", "truncated"],
)
def test_meminfo_available_declines_a_file_it_cannot_read(tmp_path: Path, text: str) -> None:
    path = tmp_path / "meminfo"
    path.write_text(text, encoding="ascii")
    assert memory._meminfo_available(str(path)) is None


def test_meminfo_available_declines_a_missing_file(tmp_path: Path) -> None:
    assert memory._meminfo_available(str(tmp_path / "absent")) is None


def test_linux_answers_with_mem_available_not_the_free_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The witness: free pages and ``MemAvailable`` differ, and the answer is the second."""
    path = tmp_path / "meminfo"
    path.write_text(MEMINFO, encoding="ascii")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(memory, "MEMINFO", str(path))
    monkeypatch.setattr(memory, "CGROUP", str(tmp_path / "no-cgroup"))
    monkeypatch.setattr(
        memory.os,
        "sysconf",
        lambda name: {"SC_AVPHYS_PAGES": 3, "SC_PAGE_SIZE": 4096}[name],
        raising=False,
    )
    assert memory.available_memory() == 9437184 * 1024


def test_linux_falls_back_to_the_free_pages_without_meminfo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not hasattr(memory.os, "sysconf"):
        pytest.skip("this platform has no sysconf")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(memory, "MEMINFO", str(tmp_path / "absent"))
    monkeypatch.setattr(memory, "CGROUP", str(tmp_path / "no-cgroup"))
    monkeypatch.setattr(
        memory.os, "sysconf", lambda name: {"SC_AVPHYS_PAGES": 3, "SC_PAGE_SIZE": 4096}[name]
    )
    assert memory.available_memory() == 3 * 4096


GIB = 1 << 30


def _group(directory: Path, files: dict[str, str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (directory / name).write_text(text, encoding="ascii")


def _cgroup_file(tmp_path: Path, text: str) -> str:
    path = tmp_path / "cgroup"
    path.write_text(text, encoding="ascii")
    return str(path)


def test_a_v2_limit_is_headroom_less_the_active_usage(tmp_path: Path) -> None:
    """``memory.current`` counts the file cache, and the headroom gives it back."""
    root = tmp_path / "fs"
    _group(
        root / "box",
        {
            "memory.max": f"{8 * GIB}\n",
            "memory.current": f"{5 * GIB}\n",
            "memory.stat": f"anon 1\ninactive_file {2 * GIB}\nactive_file {GIB}\nshmem 7\n",
        },
    )
    cgroup = _cgroup_file(tmp_path, "0::/box\n")
    assert memory._cgroup_headroom(cgroup, str(root)) == 6 * GIB


def test_the_tightest_ancestor_decides(tmp_path: Path) -> None:
    """A v1 child without a limit still sits under its parent's limit."""
    root = tmp_path / "fs"
    unlimited = str(9223372036854771712)
    _group(
        root / "memory" / "outer",
        {
            "memory.limit_in_bytes": f"{4 * GIB}",
            "memory.usage_in_bytes": f"{3 * GIB}",
            "memory.stat": "total_inactive_file 0\n",
        },
    )
    _group(
        root / "memory" / "outer" / "inner",
        {
            "memory.limit_in_bytes": unlimited,
            "memory.usage_in_bytes": f"{3 * GIB}",
            "memory.stat": "total_inactive_file 0\n",
        },
    )
    _group(root / "memory", {"memory.limit_in_bytes": unlimited, "memory.usage_in_bytes": "1"})
    cgroup = _cgroup_file(tmp_path, "4:memory:/outer/inner\n2:cpu:/elsewhere\n")
    assert memory._cgroup_headroom(cgroup, str(root)) == GIB


def test_a_path_outside_the_mount_reads_the_mount_root(tmp_path: Path) -> None:
    """A container mounts its own group as the root, so the host path does not exist there."""
    root = tmp_path / "fs"
    _group(root, {"memory.max": f"{2 * GIB}", "memory.current": f"{GIB}"})
    cgroup = _cgroup_file(tmp_path, "0::/host/side/path\n")
    assert memory._cgroup_headroom(cgroup, str(root)) == GIB


@pytest.mark.parametrize(
    ("cgroup", "files"),
    [
        ("0::/\n", {"memory.max": "max\n", "memory.current": "5"}),
        ("0::/\n", {}),
        ("3:cpu:/\n", {"memory.max": "10", "memory.current": "5"}),
    ],
    ids=["unlimited", "no-controller-files", "no-memory-controller"],
)
def test_no_limit_gives_no_headroom(tmp_path: Path, cgroup: str, files: dict[str, str]) -> None:
    root = tmp_path / "fs"
    _group(root, files)
    assert memory._cgroup_headroom(_cgroup_file(tmp_path, cgroup), str(root)) is None


def test_a_missing_cgroup_file_gives_no_headroom(tmp_path: Path) -> None:
    assert memory._cgroup_headroom(str(tmp_path / "absent"), str(tmp_path)) is None


def test_linux_caps_mem_available_by_the_cgroup_headroom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The witness: the host reports 9 GiB, the group leaves 1 GiB, and the answer is 1 GiB."""
    path = tmp_path / "meminfo"
    path.write_text(MEMINFO, encoding="ascii")
    root = tmp_path / "fs"
    _group(root, {"memory.max": f"{2 * GIB}", "memory.current": f"{GIB}"})
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(memory, "MEMINFO", str(path))
    monkeypatch.setattr(memory, "CGROUP", _cgroup_file(tmp_path, "0::/\n"))
    monkeypatch.setattr(memory, "CGROUP_ROOT", str(root))
    assert memory.available_memory() == GIB
    # Without a limit the host count stands.
    (root / "memory.max").write_text("max", encoding="ascii")
    assert memory.available_memory() == 9437184 * 1024
