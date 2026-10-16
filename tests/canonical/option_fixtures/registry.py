"""The registry of the option fixtures: one R script and one JSON file each.

Each fixture holds the R side of a comparison for a targeting or fitting option: the ``tmle3``
updater (parts a to c), ``drtmle(stratify=)`` (part f), ``ctmleDiscrete(patience=)`` (part g) and
``ctmleDiscrete(Wg=)`` (part h).  R is not part of the cloud session that wrote the scripts, so every fixture ships ``pending``.
A pending fixture has no JSON file, a reason, and the checksum of its script.  The local Study queue runs the script, adds the JSON and sets the status to
``present`` in a follow-up change.  :func:`check_entry` reads the state of one entry and returns
the findings; ``tests/unit/test_option_fixtures.py`` asserts that none are left.

A missing comparator never blocks a method (``docs/architecture-invariants.md``), and the Python
exact tests of each option gate the change.  The pending state is asserted and reported, never
skipped without a message and never read as a pass.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "DOCS_FILES",
    "HERE",
    "PENDING_LABEL",
    "REGISTRY",
    "ROOT",
    "FixtureEntry",
    "build_command",
    "check_entry",
    "dockerfile_facts",
    "label_findings",
    "load",
    "run_command",
    "script_sha256",
]

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

#: The label a reader-facing evidence row carries while its comparator run is pending.
PENDING_LABEL = "exact tests, with comparator runs pending"

#: The two reader-facing pages whose rows name a fixture file.
DOCS_FILES = ("docs/technical-reference/evidence.md", "docs/technical-reference/index.md")

#: Why every fixture is pending today.
_REASON = (
    "R is not installed in the cloud session that wrote the script; the run waits in the local "
    "Study queue (request X22-R-fixtures)"
)

_STATUSES = ("pending", "present")


@dataclass(frozen=True)
class FixtureEntry:
    """One fixture: the script that makes it, the image that runs it, and its state.

    Parameters
    ----------
    name : str
        The fixture name.  The JSON file is ``<name>.json``.
    parts : tuple of str
        The option parts the fixture checks.
    comparator : str
        The R call the script makes, in words.
    script : str
        The script file, in this directory.
    image : str
        The tag of the pinned image that runs the script.
    context : str
        The directory, relative to the repository root, whose Dockerfile builds the image.
    packages : tuple of str
        The R packages the script reports in its provenance.
    status : {"pending", "present"}
        ``"pending"`` when the JSON does not exist yet.
    reason : str
        Why the fixture is pending.  Empty for a present fixture.
    script_sha256 : str
        The SHA-256 of the script bytes, which are LF only.
    """

    name: str
    parts: tuple[str, ...]
    comparator: str
    script: str
    image: str
    context: str
    packages: tuple[str, ...]
    status: str
    reason: str
    script_sha256: str

    @property
    def output(self) -> str:
        """The JSON file name."""
        return f"{self.name}.json"

    @property
    def pending(self) -> bool:
        """Whether the JSON is still to come."""
        return self.status == "pending"


REGISTRY: tuple[FixtureEntry, ...] = (
    FixtureEntry(
        name="tmle3_updater",
        parts=("a", "b", "c"),
        comparator="tmle_TSM_all() with tmle3_Update: sample_size, scaled_var and constrain_step",
        script="tmle3_updater.R",
        image="cleverly-tmle3-reference:ed72f8a",
        context="tests/canonical/tmle3",
        packages=("tmle3", "sl3"),
        status="pending",
        reason=_REASON,
        script_sha256="479960e47e90c756b39867d293c5590afd2b46dfbe26d2e1170b632fa4e0b236",
    ),
    FixtureEntry(
        name="drtmle_stratify",
        parts=("f",),
        comparator="drtmle(stratify = TRUE) and drtmle(stratify = FALSE), guard = NULL",
        script="drtmle_stratify.R",
        image="cleverly-drtmle-reference:538a3a2",
        context="tests/canonical/drtmle",
        packages=("drtmle",),
        status="pending",
        reason=_REASON,
        script_sha256="784c88934bee039a6e67edcc123c087dcdc3ef3863599bf4c50992262be7f9fa",
    ),
    FixtureEntry(
        name="ctmle_patience",
        parts=("g",),
        comparator="ctmleDiscrete(preOrder = TRUE, patience = 2) and patience = FALSE",
        script="ctmle_patience.R",
        image="cleverly-ctmle-reference:18de559",
        context="tests/canonical/ctmle_selector",
        packages=("ctmle",),
        status="pending",
        reason=_REASON,
        script_sha256="a66fb36987fb3caea38ab9ec3f3b233534230918350d1a23a82b9bf6175f423c",
    ),
    FixtureEntry(
        name="ctmle_wg",
        parts=("h",),
        comparator="ctmleDiscrete(Wg = W[, c(1, 3)], preOrder = TRUE) and Wg = W",
        script="ctmle_wg.R",
        image="cleverly-ctmle-reference:18de559",
        context="tests/canonical/ctmle_selector",
        packages=("ctmle",),
        status="pending",
        reason=_REASON,
        script_sha256="b24782f31c47fd5f1652e79314cb41e7c9acdf64c692f1bceaeb55b72aa207a2",
    ),
)


def script_sha256(path: Path) -> str:
    """The SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_command(entry: FixtureEntry) -> str:
    """The command that builds the pinned image, run from the repository root."""
    return f"docker build -t {entry.image} {entry.context}"


def run_command(entry: FixtureEntry) -> str:
    """The command the local Study queue runs for one fixture, from the repository root.

    The container mounts this directory, runs the script with ``Rscript`` (the ``ctmle`` image
    sets another entry point) and writes the JSON file beside the script.  The image id and the
    script checksum enter the provenance of the file.
    """
    mount = "$PWD/tests/canonical/option_fixtures"
    return (
        "docker run --rm --entrypoint Rscript "
        f"-e CLEVERLY_IMAGE_ID=\"$(docker image inspect --format '{{{{.Id}}}}' {entry.image})\" "
        f'-e CLEVERLY_SCRIPT_SHA256="$(sha256sum {mount}/{entry.script} | cut -d" " -f1)" '
        f'-v "{mount}:/fixture" {entry.image} /fixture/{entry.script} /fixture/{entry.output}'
    )


def dockerfile_facts(entry: FixtureEntry, root: Path = ROOT) -> dict[str, str]:
    """The base image digest and the package commit of the Dockerfile that builds the image.

    Parameters
    ----------
    entry : FixtureEntry
        The fixture.
    root : pathlib.Path
        The repository root.

    Returns
    -------
    dict
        ``base`` is the ``FROM`` reference with its ``sha256`` digest, and ``commit`` is the
        40-character commit the Dockerfile installs the R package at.
    """
    text = (root / entry.context / "Dockerfile").read_text(encoding="utf-8")
    base = re.search(r"^FROM\s+(\S+@sha256:[0-9a-f]{64})\s*$", text, flags=re.MULTILINE)
    commit = re.search(r"^ARG\s+[A-Z0-9]+_COMMIT=([0-9a-f]{40})\s*$", text, flags=re.MULTILINE)
    return {
        "base": base.group(1) if base else "",
        "commit": commit.group(1) if commit else "",
    }


def load(entry: FixtureEntry, root: Path = HERE) -> dict[str, Any] | None:
    """The fixture's JSON, or ``None`` while the file does not exist."""
    path = root / entry.output
    if not path.exists():
        return None
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def check_entry(entry: FixtureEntry, root: Path = HERE) -> list[str]:
    """The findings on the state of one entry.

    A pending entry has a reason, no JSON, and a script whose bytes are LF only and match the
    registered checksum.  A present entry has the JSON, which names the fixture and records the
    same script checksum.

    Parameters
    ----------
    entry : FixtureEntry
        The registry entry.
    root : pathlib.Path
        The directory holding the script and the JSON.

    Returns
    -------
    list of str
        One sentence per finding.  An empty list is a coherent entry.
    """
    findings: list[str] = []
    if entry.status not in _STATUSES:
        return [f"{entry.name}: status {entry.status!r} is neither 'pending' nor 'present'"]
    script = root / entry.script
    if not script.exists():
        return [f"{entry.name}: the script {entry.script} does not exist"]
    data = script.read_bytes()
    if b"\r" in data:
        findings.append(f"{entry.name}: {entry.script} holds CR bytes; the script must be LF only")
    actual = hashlib.sha256(data).hexdigest()
    if actual != entry.script_sha256:
        findings.append(
            f"{entry.name}: the registered checksum {entry.script_sha256[:12]}... is not the "
            f"checksum of {entry.script} ({actual[:12]}...)"
        )
    path = root / entry.output
    if entry.pending:
        if not entry.reason.strip():
            findings.append(f"{entry.name}: a pending fixture needs a reason")
        if path.exists():
            findings.append(f"{entry.name}: pending, but {entry.output} exists; set it present")
        return findings
    if entry.reason.strip():
        findings.append(f"{entry.name}: a present fixture carries no pending reason")
    if not path.exists():
        findings.append(f"{entry.name}: present, but {entry.output} does not exist")
        return findings
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("fixture") != entry.name:
        findings.append(f"{entry.name}: {entry.output} names fixture {payload.get('fixture')!r}")
    recorded = (payload.get("provenance") or {}).get("script_sha256")
    if recorded != entry.script_sha256:
        findings.append(
            f"{entry.name}: {entry.output} records script checksum {recorded!r}, not the "
            "registered one; the file came from another version of the script"
        )
    if not (payload.get("provenance") or {}).get("image_id"):
        findings.append(f"{entry.name}: {entry.output} records no image id")
    return findings


def label_findings(
    entries: Iterable[FixtureEntry], texts: Mapping[str, str], label: str = PENDING_LABEL
) -> list[str]:
    """The findings on the pending label of the reader-facing rows.

    A row is a Markdown table line that names a fixture file in backticks.  The row carries the
    label exactly when the fixture is pending.  Once any row names a fixture, every page that
    names one names all of them.

    Parameters
    ----------
    entries : iterable of FixtureEntry
        The registry, or a modified copy of it.
    texts : mapping of str to str
        The page text by page name.
    label : str
        The label a pending row carries.

    Returns
    -------
    list of str
        One sentence per finding.
    """
    entries = tuple(entries)
    findings: list[str] = []
    for page, text in texts.items():
        rows = [line for line in text.splitlines() if line.lstrip().startswith("|")]
        named = {
            entry.name: [row for row in rows if f"`{entry.output}`" in row] for entry in entries
        }
        if not any(named.values()):
            continue
        for entry in entries:
            matching = named[entry.name]
            if not matching:
                findings.append(f"{page}: no row names {entry.output}")
            for row in matching:
                carries = label in row
                if entry.pending and not carries:
                    findings.append(
                        f"{page}: the row for pending {entry.output} lacks the label {label!r}"
                    )
                if not entry.pending and carries:
                    findings.append(
                        f"{page}: the row for present {entry.output} still carries the pending label"
                    )
    return findings


if __name__ == "__main__":
    for item in REGISTRY:
        print(f"{item.name} {script_sha256(HERE / item.script)}")
