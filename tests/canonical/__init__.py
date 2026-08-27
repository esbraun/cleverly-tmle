"""Frozen cross-implementation evidence fixtures."""

from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: A file a runner reads into its own environment before it runs.  The container binds
#: this directory at ``/fixture``, which is what the path resolves against.
_SOURCED = re.compile(r"""source\(["'](?P<path>/fixture/[^"']+)["']\)""")


def runner_source(runner: Path) -> str:
    """One reference runner's text, plus every file it sources.

    A test that asks what a runner does has to read the runner *as executed*.  Half of
    each one now lives in ``study_harness.R`` and, for the regimen studies, in
    ``ltmle_regimen_adapter.R``; a check written against the file alone would pass or fail
    on where a line happens to sit rather than on what the run does.
    """
    text = runner.read_text(encoding="utf-8")
    parts = [text]
    for match in _SOURCED.finditer(text):
        sourced = HERE / match.group("path").removeprefix("/fixture/")
        if not sourced.exists():  # pragma: no cover - a study contract failure
            raise AssertionError(
                f"{runner.parent.name} sources {match.group('path')}, which is not here"
            )
        parts.append(sourced.read_text(encoding="utf-8"))
    return "\n".join(parts)
