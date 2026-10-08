"""The physical memory available to this process, read with the standard library.

``psutil`` is not a dependency.  Windows answers through ``GlobalMemoryStatusEx``, and a
POSIX system through its available-page count.  macOS has no available-page count, so the
answer there is ``None``, and a caller that guards an allocation then skips its guard.
"""

from __future__ import annotations

import os

__all__ = ["available_memory"]


def available_memory() -> int | None:
    """The physical memory available now, in bytes, or ``None`` where it cannot be read.

    Returns
    -------
    int or None
        The available bytes, or ``None`` on a platform that does not report them.
    """
    if os.name == "nt":
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
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return None
        return int(status.ullAvailPhys)
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None
