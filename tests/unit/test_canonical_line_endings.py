"""No published artefact may carry a CRLF, including inside a compressed one.

``tests/studies/evidence/manifest.py`` records a sha256 over every published artefact and
``test_method_evidence.py`` checks it, so the bytes have to be a property of the study rather
than of the machine that wrote it.  ``pandas.to_csv`` and :meth:`pathlib.Path.write_text` both
default to ``os.linesep``, which is CRLF on Windows, and the module answers that with
:data:`~tests.studies.evidence.manifest.NEWLINE` on every write.

``.gitattributes`` answers the other half, the checkout.  Between them the uncompressed tables
are covered twice: a stray CRLF in one is visible in a diff and normalised on the way in.

**The archives are covered by neither.**  Git treats ``*.gz`` as binary and never looks inside,
and a CRLF written *before* the compression is invisible to every other check in this
repository -- to a diff, to the checkout filter, and to the manifest, which faithfully hashes
the contaminated bytes.  Three committed archives were found carrying one, and the only reason
it surfaced at all is that a regeneration happened to produce the corrected file beside the
original.

So this reads the bytes.  It is the cheapest gate in the tree and the only one that can see
this class of defect at all.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from tests.studies.evidence.registry import ROOT

CANONICAL = ROOT / "tests" / "canonical"

#: Every committed result file, compressed or not, plus the manifests beside them.
ARTIFACTS = tuple(
    sorted(
        path
        for pattern in ("*/*.csv", "*/*.csv.gz", "*/manifest.json")
        for path in CANONICAL.glob(pattern)
    )
)


def published_bytes(path: Path) -> bytes:
    return gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()


def test_every_study_directory_is_covered() -> None:
    """A new study joins this check by existing rather than by being listed."""
    covered = {path.parent.name for path in ARTIFACTS}
    directories = {path.name for path in CANONICAL.iterdir() if (path / "manifest.json").exists()}
    assert directories <= covered, f"no artefact found for {sorted(directories - covered)}"


@pytest.mark.parametrize("artifact", ARTIFACTS, ids=lambda path: f"{path.parent.name}/{path.name}")
def test_no_published_artefact_carries_a_carriage_return(artifact: Path) -> None:
    content = published_bytes(artifact)
    count = content.count(b"\r\n")
    assert count == 0, (
        f"{artifact.relative_to(ROOT).as_posix()} carries {count} CRLF line ending(s). "
        f"Its recorded sha256 then verifies on the machine that wrote it and nowhere else. "
        f"Regenerate the study; tests/studies/evidence/manifest.py writes LF"
    )
