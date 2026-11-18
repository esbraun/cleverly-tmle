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

import operator
import os
import sys
from typing import Any

__all__ = ["available_memory", "check_max_memory_bytes", "check_memory_budget"]

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


def check_max_memory_bytes(value: Any, *, name: str = "max_memory_bytes") -> int | None:
    """A byte-budget argument as a positive number of bytes, or ``None``.

    The caller decides what ``None`` means: a memory guard reads it as the memory available
    now, and the assessment cache's ``max_bytes`` as no budget.

    Parameters
    ----------
    value : int or None
        The argument: ``None``, or a positive integer (a NumPy integer too, but not a
        ``bool``).
    name : str, default="max_memory_bytes"
        The argument's name, as the error states it.

    Returns
    -------
    int or None
        The budget in bytes, or ``None``.

    Raises
    ------
    DataError
        For any other value.
    """
    from ..exceptions import DataError

    if value is None:
        return None
    message = f"{name} must be a positive integer or None; got {value!r}"
    if isinstance(value, bool):
        raise DataError(message)
    try:
        budget = operator.index(value)
    except TypeError:
        raise DataError(message) from None
    if budget <= 0:
        raise DataError(message)
    return int(budget)


def check_memory_budget(
    predicted: int, *, max_memory_bytes: int | None, what: str, factor: float
) -> int | None:
    """Refuse an allocation predicted to exceed the memory budget, before it is made.

    The budget is ``max_memory_bytes`` when given, in either direction, and otherwise the
    memory available now; where the platform reports none, nothing is checked.  The
    prediction counts what the guarded stage adds above the arrays already held, so the
    budget is not a cap on the process.

    Parameters
    ----------
    predicted : int
        The predicted peak bytes of the guarded stage.
    max_memory_bytes : int or None
        A budget already checked by :func:`check_max_memory_bytes`, or ``None``.
    what : str
        The stage, as the refusal names it.
    factor : float
        How far the prediction may exceed the true peak, which the refusal states.

    Returns
    -------
    int or None
        The budget the check used, or ``None`` when none applies.

    Raises
    ------
    MethodConfigurationError
        When the prediction exceeds the budget.
    """
    from ..exceptions import MethodConfigurationError, RefusalReason

    budget = available_memory() if max_memory_bytes is None else max_memory_bytes
    if budget is None or predicted <= budget:
        return budget

    def size(count: int) -> str:
        return f"{count / 1024.0**2:,.1f} MiB" if count >= 1024**2 else f"{count:,} bytes"

    source = (
        "is available now" if max_memory_bytes is None else "is the budget max_memory_bytes sets"
    )
    raise MethodConfigurationError(
        f"{what} would need about {size(predicted)} above the arrays already held, and "
        f"{size(budget)} {source}. The prediction may exceed the true peak by up to "
        f"a factor of {factor:g}, and the available memory is read once, before the stage "
        "starts, so a request that would just fit can be refused.",
        reason=RefusalReason.INVALID_DECLARATION,
        alternative="Pass max_memory_bytes= (in bytes) to set the budget yourself.",
    )
