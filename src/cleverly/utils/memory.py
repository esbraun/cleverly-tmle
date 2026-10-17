"""The physical memory available to this process, read with the standard library.

``psutil`` is not a dependency.  Windows answers through ``GlobalMemoryStatusEx``, and a
Linux system through ``MemAvailable`` in ``/proc/meminfo``, and another POSIX system through
its free-page count.  macOS has no free-page count, so the answer there is ``None``, and a
caller that guards an allocation then skips its guard.

Linux reads ``MemAvailable`` and not the free-page count because the kernel reclaims the page
cache on demand.  After a long run much of the memory can sit in that cache, and the free-page
count then understates what an allocation can use.

On Linux the answer is also capped by the memory limit of the process's control groups
(cgroup v2 ``memory.max``, or cgroup v1 ``memory.limit_in_bytes``). A container sees the memory
of the host in ``/proc/meminfo``, and a pool that sized itself by that count could exceed the
container limit and lose a worker to the out-of-memory killer. The headroom of a group is its
limit less its usage, and the usage leaves out the file cache (the active and inactive file
pages), which the kernel reclaims before it kills a process.
"""

from __future__ import annotations

import os
import sys

__all__ = ["available_memory"]

#: The Linux file that :func:`available_memory` reads ``MemAvailable`` from.
MEMINFO = "/proc/meminfo"

#: The Linux file that names the control groups of this process.
CGROUP = "/proc/self/cgroup"

#: Where Linux mounts the control group file systems.
CGROUP_ROOT = "/sys/fs/cgroup"

#: Per cgroup version: the limit, the usage, and the ``memory.stat`` keys of the file cache.
#: cgroup v1 reports a missing limit as a huge number, and v2 as ``max``.
_CGROUP_FILES = {
    2: ("memory.max", "memory.current", ("inactive_file", "active_file")),
    1: (
        "memory.limit_in_bytes",
        "memory.usage_in_bytes",
        ("total_inactive_file", "total_active_file"),
    ),
}

#: A cgroup v1 limit at or above this many bytes means no limit (the kernel writes a page-rounded
#: ``LLONG_MAX``).
_V1_UNLIMITED = 1 << 60


def _meminfo_available(path: str) -> int | None:
    """``MemAvailable`` from a ``/proc/meminfo`` file, in bytes, or ``None`` without one."""
    try:
        with open(path, encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    value, unit = line.split()[1:3]
                    if unit != "kB":
                        return None
                    return int(value) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _read_int(path: str) -> int | None:
    """One integer from a control group file, or ``None`` for ``max``, a gap or a bad file."""
    try:
        with open(path, encoding="ascii") as handle:
            text = handle.read().strip()
    except OSError:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _stat_value(path: str, key: str) -> int:
    """A ``memory.stat`` entry in bytes, or 0 without one."""
    try:
        with open(path, encoding="ascii") as handle:
            for line in handle:
                name, _, value = line.partition(" ")
                if name == key:
                    return int(value)
    except (OSError, ValueError):
        return 0
    return 0


def _cgroup_directories(cgroup: str, root: str) -> list[tuple[int, str]]:
    """The memory control group of this process and its ancestors, with their versions.

    A container usually mounts its own group as the root, and ``/proc/self/cgroup`` then names
    a path that does not exist under the mount.  The mount root is then the group.
    """
    try:
        with open(cgroup, encoding="ascii") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return []
    found: list[tuple[int, str]] = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, path = parts
        if controllers == "":
            version, mount = 2, root
            if os.path.isdir(os.path.join(root, "unified")):
                mount = os.path.join(root, "unified")
        elif "memory" in controllers.split(","):
            version, mount = 1, os.path.join(root, "memory")
        else:
            continue
        relative = path.strip("/")
        if not os.path.isdir(os.path.join(mount, relative)):
            relative = ""
        while True:
            found.append((version, os.path.join(mount, relative) if relative else mount))
            if not relative:
                break
            relative = os.path.dirname(relative)
    return found


def _cgroup_headroom(cgroup: str, root: str) -> int | None:
    """The smallest memory headroom among the limited control groups, or ``None`` without one."""
    headroom: int | None = None
    for version, directory in _cgroup_directories(cgroup, root):
        limit_name, usage_name, cache_keys = _CGROUP_FILES[version]
        limit = _read_int(os.path.join(directory, limit_name))
        if limit is None or (version == 1 and limit >= _V1_UNLIMITED):
            continue
        usage = _read_int(os.path.join(directory, usage_name))
        if usage is None:
            continue
        stat = os.path.join(directory, "memory.stat")
        cache = sum(_stat_value(stat, key) for key in cache_keys)
        room = max(0, limit - max(0, usage - cache))
        headroom = room if headroom is None else min(headroom, room)
    return headroom


def _linux_available() -> int | None:
    """``MemAvailable``, capped by the control group headroom; ``None`` without either."""
    reclaimable = _meminfo_available(MEMINFO)
    headroom = _cgroup_headroom(CGROUP, CGROUP_ROOT)
    if reclaimable is None:
        return headroom
    return reclaimable if headroom is None else min(reclaimable, headroom)


def available_memory() -> int | None:
    """The physical memory available now, in bytes, or ``None`` where it cannot be read.

    Returns
    -------
    int or None
        The available bytes, or ``None`` on a platform that does not report them.
    """
    if sys.platform == "win32":
        import ctypes

        class _Status(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _Status()
        status.dwLength = ctypes.sizeof(_Status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.ullAvailPhys)
    if sys.platform.startswith("linux"):
        linux = _linux_available()
        if linux is not None:
            return linux
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None
