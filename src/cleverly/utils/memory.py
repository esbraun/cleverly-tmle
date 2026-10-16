"""The physical memory available to this process, read with the standard library.

``psutil`` is not a dependency.  Windows answers through ``GlobalMemoryStatusEx``, and a
Linux system through ``MemAvailable`` in ``/proc/meminfo``, and another POSIX system through
its free-page count.  macOS has no free-page count, so the answer there is ``None``, and a
caller that guards an allocation then skips its guard.

Linux reads ``MemAvailable`` and not the free-page count because the kernel reclaims the page
cache on demand.  After a long run much of the memory can sit in that cache, and the free-page
count then understates what an allocation can use.

Neither count reads a container's cgroup limit. Both report the memory of the host.
"""

from __future__ import annotations

import os
import sys

__all__ = ["available_memory"]

#: The Linux file that :func:`available_memory` reads ``MemAvailable`` from.
MEMINFO = "/proc/meminfo"


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
        reclaimable = _meminfo_available(MEMINFO)
        if reclaimable is not None:
            return reclaimable
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None
