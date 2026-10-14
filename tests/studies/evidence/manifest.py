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

#: The one line ending a published artefact may contain.
#:
#: These bytes are hashed below and the hash is committed, so they have to be a property of
#: the study rather than of the machine that ran it.  ``pandas.to_csv`` and
#: :meth:`pathlib.Path.write_text` both default to ``os.linesep``, which is CRLF on Windows;
#: git then stores LF in the blob, so the recorded digest verifies on the machine that wrote
#: it and nowhere else.  ``.gitattributes`` pins the checkout side of the same agreement.
#: The uncompressed tables are the ones that can break -- git leaves ``.csv.gz`` alone as
#: binary -- but the archives are written the same way so regenerating on two platforms
#: produces one answer rather than two that differ invisibly.
NEWLINE = "\n"


def hashes(paths: Iterable[Path]) -> dict[str, str]:
    """Digest each published artefact, keyed by the name it has in the study's directory."""
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def reference_hashes(paths: Iterable[Path]) -> dict[str, str]:
    """Digest each reference source, keyed by its path from the repository root.

    Not :func:`hashes`, and the difference is not cosmetic.  An artefact is *in* the study's
    directory, so its bare name locates it.  A reference source need not be: two studies can
    share one Docker context and one sourced adapter, which is what
    :attr:`~tests.canonical.regenerate.Reference.build_context` and ``runner_root`` exist for.
    Keyed by bare name, those shared files resolve to paths that do not exist under either
    study, and the manifest check reads as a missing file rather than as a shared one.  A
    repository-relative key locates both cases, which is what ``study_module_sha256`` already
    does for the same reason.
    """
    return {
        path.resolve().relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def write_csv(frame: Any, path: Path, **options: Any) -> None:
    """Write one published table with the line ending :data:`NEWLINE` fixes."""
    frame.to_csv(path, index=False, lineterminator=NEWLINE, **options)


def write_lines(path: Path, text: str) -> None:
    """Write one published document with the line ending :data:`NEWLINE` fixes."""
    path.write_text(text, encoding="utf-8", newline=NEWLINE)


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
    """Write the artefact manifest for one study.

    A record whose comparator is pending (:attr:`StudyRecord.comparator_state`) records no
    reference provenance and an empty ``reference_sha256``, because no reference ran.  Its
    ``comparator`` block names the pending implementation, the reason, the declared reference
    metadata, and the digest of each reference source as declared at this run
    (``declared_sources_sha256``).  No test gates those digests: a comparator fix made while
    the runs are pending changes bytes that never ran.
    """
    pending = record.comparator_state == "pending"
    files = tuple(reference_files)
    payload: dict[str, Any] = {
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
            **({"scenario_n": dict(record.scenario_n)} if record.scenario_n else {}),
            "seed": record.seed,
            **(
                {"resampling_seed": record.resampling_seed}
                if record.resampling_seed is not None
                else {}
            ),
            **(
                {"scenario_seed_owners": dict(record.scenario_seed_owners)}
                if record.scenario_seed_owners
                else {}
            ),
            "publication_policy": record.publication_policy,
            **({"truth_varies_by_replicate": True} if record.truth_varies_by_replicate else {}),
            **({"failure_bound": dict(record.failure_bound)} if record.failure_bound else {}),
            **(
                {"accepted_reference_failure": record.accepted_reference_failure}
                if record.accepted_reference_failure
                else {}
            ),
            "scenarios": {name: list(names) for name, names in record.scenarios.items()},
            **dict(configuration or {}),
            "margins": record.margins.as_json(),
        },
        "study_module_sha256": study_module_hashes(record),
        "reference_sha256": {} if pending else reference_hashes(files),
        "sha256": hashes(artifacts),
    }
    if pending:
        payload["comparator"] = {
            "status": "pending",
            "implementation": record.pending_reference,
            "reason": record.pending_reason,
            "metadata": dict(reference_metadata or {}),
            "declared_sources_sha256": reference_hashes(files),
        }
    write_lines(path, json.dumps(payload, indent=2) + "\n")


def comparator_findings(
    record: StudyRecord,
    manifest: Mapping[str, Any],
    implementations: Iterable[str],
    paired_rows: int,
) -> list[str]:
    """Every way the manifest and the published rows disagree with the record's comparator state.

    ``implementations`` are the names in ``replicates.csv.gz``, and ``paired_rows`` is the row
    count of ``equivalence.csv``.  An empty list means they agree.  A pending comparator is an
    asserted state, not a skip: its manifest must say pending, it records no reference
    provenance, and no row of the pending implementation and no paired row is published.  An
    empty paired table would otherwise read as every paired verdict passed.
    """
    found: list[str] = []
    published = set(implementations)
    comparator = manifest.get("comparator")
    reference = manifest["generated_with"].get("reference")
    state = record.comparator_state
    if state != "pending":
        if comparator is not None:
            found.append(
                f"the record's comparator is {state}, but the manifest records one as pending"
            )
        if (reference is None) != (state == "none"):
            found.append(
                f"the record's comparator is {state}, but the manifest's reference is {reference!r}"
            )
        expected = set(record.implementations)
        if published != expected:
            found.append(f"the rows publish {sorted(published)}, not {sorted(expected)}")
        if state == "none" and paired_rows:
            found.append(
                f"no comparator is declared, but {paired_rows} paired row(s) are published"
            )
        return found
    if comparator is None or comparator.get("status") != "pending":
        found.append("the record's comparator is pending, but the manifest does not say so")
        return found
    if comparator.get("implementation") != record.pending_reference:
        found.append(
            f"the manifest's pending comparator is {comparator.get('implementation')!r}, "
            f"not {record.pending_reference!r}"
        )
    if comparator.get("reason") != record.pending_reason:
        found.append("the manifest's pending reason is not the record's")
    if not comparator.get("declared_sources_sha256"):
        found.append("the pending manifest records no declared reference source")
    if reference is not None:
        found.append("a pending comparator has reference provenance, but no reference ran")
    if manifest.get("reference_sha256"):
        found.append("a pending comparator hashes reference sources as run, but none ran")
    if record.pending_reference in published:
        found.append(f"rows of the pending comparator {record.pending_reference} are published")
    if published != {record.implementation}:
        found.append(f"the rows publish {sorted(published)}, not only {record.implementation}")
    if paired_rows:
        found.append(f"{paired_rows} paired row(s) are published while the comparator is pending")
    return found
