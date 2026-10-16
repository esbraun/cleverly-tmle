"""Linux reads the reclaimable memory, not the free pages.

A CI runner refused a 0.8 GiB density design with 0.6 GiB "available" late in the fast suite.
The free-page count it read leaves out the page cache, which the kernel gives back on demand.
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
    monkeypatch.setattr(memory._meminfo_available, "__defaults__", (str(path),))
    assert memory.available_memory() == 9437184 * 1024


def test_linux_falls_back_to_the_free_pages_without_meminfo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not hasattr(memory.os, "sysconf"):
        pytest.skip("this platform has no sysconf")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(memory._meminfo_available, "__defaults__", (str(tmp_path / "absent"),))
    monkeypatch.setattr(
        memory.os, "sysconf", lambda name: {"SC_AVPHYS_PAGES": 3, "SC_PAGE_SIZE": 4096}[name]
    )
    assert memory.available_memory() == 3 * 4096
