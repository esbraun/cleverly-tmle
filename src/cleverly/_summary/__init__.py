"""The printed summary every result object shares.

Users meet this module through ``result.summary()`` only. It is private so a seam can
change a layout without a public deprecation.
"""

from __future__ import annotations

from typing import Literal

__all__ = ["Detail", "check_detail"]

#: How much a summary prints: the analysis (``"short"``, the default), or the analysis with
#: its assumptions, settings and study description (``"full"``).
Detail = Literal["short", "full"]


def check_detail(detail: str) -> Detail:
    """Return ``detail`` when it names a detail level, and raise otherwise.

    Parameters
    ----------
    detail : str
        The ``detail=`` argument a ``summary()`` received.

    Returns
    -------
    {"short", "full"}
        The same value.

    Raises
    ------
    ValueError
        When ``detail`` is neither ``"short"`` nor ``"full"``.
    """
    if detail not in ("short", "full"):
        raise ValueError(f"detail must be 'short' or 'full'; got {detail!r}")
    return detail  # type: ignore[return-value]
