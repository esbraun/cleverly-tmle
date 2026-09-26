"""The one cross-version rule for saved objects: warn on a version mismatch, then load.

``cleverly`` is alpha, and it keeps no compatibility for saved objects. A saved result
records the version that wrote it, and a load by any other version warns with
:class:`~cleverly.exceptions.VersionMismatchWarning`. The load then proceeds as saved, with
no migration, backfill, or re-stamp. This module imports only the version and the
exception types, so every loader can reach it.
"""

from __future__ import annotations

import warnings

from ._version import __version__
from .exceptions import VersionMismatchWarning

__all__ = ["warn_on_version_mismatch"]


def warn_on_version_mismatch(
    saved: str | None, *, what: str = "result", stacklevel: int = 3
) -> None:
    """Warn when a saved object was written by a different cleverly version.

    Parameters
    ----------
    saved : str or None
        The version the saved object recorded, or ``None`` when it recorded no version.
    what : str, default "result"
        The kind of saved object, named in the warning.
    stacklevel : int, default 3
        Passed to :func:`warnings.warn`, so the warning names the caller's load call.

    Warns
    -----
    VersionMismatchWarning
        When ``saved`` differs from :data:`cleverly.__version__`, or is ``None``.
    """
    if saved == __version__:
        return
    by = f"cleverly {saved}" if saved else "an earlier cleverly release that recorded no version"
    warnings.warn(
        f"this {what} was saved by {by}, and this is cleverly {__version__}. Alpha releases "
        "keep no compatibility for saved objects, so cleverly loads it as saved, with no "
        "migration. Its statuses, reports and cached assessments can differ from what this "
        "version computes. Fit the analysis again with this version.",
        VersionMismatchWarning,
        stacklevel=stacklevel,
    )
