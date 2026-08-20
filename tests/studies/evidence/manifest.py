"""What a published artefact set says about where it came from.

The reference side of a comparison is pinned by construction -- a container digest and two
package commits -- so recording it is easy and it was recorded.  The subject side is
whatever was in the working tree, and writing down the literal words "working tree" records
nothing: it cannot distinguish the run that produced these numbers from any later state of
the repository.  What makes the artefacts non-stale is re-execution, which the study's own
tests do; what this file adds is the identification a reader needs to reproduce a run and a
bisecting developer needs to place one.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tests.studies.evidence.registry import ROOT, StudyRecord

UNKNOWN = "unknown"


def hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _git(*arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN
    return completed.stdout.strip()


def provenance() -> dict[str, Any]:
    """Version and revision of everything on the subject side of the comparison."""
    import platform

    import numpy
    import pandas
    import scipy
    import sklearn

    import cleverly

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "cleverly_version": cleverly.__version__,
        "cleverly_commit": commit,
        "cleverly_worktree_clean": status == "" if status != UNKNOWN else None,
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
    }


def study_module_hashes(record: StudyRecord) -> dict[str, str]:
    """The study's own modules, so a reader can tell which analysis wrote these numbers.

    Not asserted equal to the working tree by any test.  The equality that matters is that
    re-running the study reproduces the rows, which the study's re-execution tests check
    directly; requiring a hash match instead would only mean the manifest had to be rewritten
    for a comment change.
    """
    return {
        module: hashlib.sha256((ROOT / module).read_bytes()).hexdigest()
        for module in record.modules
    }


def write_manifest(
    path: Path,
    record: StudyRecord,
    artifacts: Iterable[Path],
    *,
    reference_files: Iterable[Path] = (),
    reference_metadata: Mapping[str, Any] | None = None,
    configuration: Mapping[str, Any] | None = None,
) -> None:
    """Write the artefact manifest for one study."""
    payload = {
        "schema_version": 2,
        "study": record.name,
        "slug": record.slug,
        "generated_with": {
            "subject": {"implementation": record.implementation, **provenance()},
            "reference": (
                None
                if record.reference is None
                else {"implementation": record.reference, **dict(reference_metadata or {})}
            ),
        },
        "configuration": {
            "replicates": record.replicates,
            "n": record.n,
            "seed": record.seed,
            "scenarios": {name: list(names) for name, names in record.scenarios.items()},
            **dict(configuration or {}),
            "margins": record.margins.as_json(),
        },
        "study_module_sha256": study_module_hashes(record),
        "reference_sha256": hashes(reference_files),
        "sha256": hashes(artifacts),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
