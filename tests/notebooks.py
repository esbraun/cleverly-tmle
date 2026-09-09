"""What a committed notebook artifact is stamped with, and which half of it is gated.

:mod:`scripts.execute_notebook` executes a notebook and writes the stamp this module builds.
The fast tier reads the same stamp back for every notebook in :data:`tests.documents.NOTEBOOKS`.
The helpers live here rather than in the script because verification needs only the digests,
and the script also imports :mod:`nbclient`, which no fast test has any use for.

**The stamp has two halves and the JSON says which is which.**  The keys in
:data:`GATED_DIGESTS` sit under ``"gated"`` and the keys in :data:`RECORDED_DIGESTS` sit under
``"recorded"``.  A flat mapping with the split declared only in the test would put the rule
somewhere a reader of the artifact never looks, and the artifact is what a reader has.

A **gated** digest covers the notebook and nothing else, so recomputing it needs no network,
no kernel, and no fitted estimator.  Whatever moves it is repairable from the file in front of
you.  The fast tier asserts these equal.

A **recorded** digest identifies the checkout that ran the notebook: the shipped ``cleverly``
source tree, the dependency lock, and the generator modules.  The fast tier asserts these are
present and well formed, and does not assert them equal.  Equality here would fail the default
handoff gate on any one-character library edit, and the only repair would be a networked
re-execution that refits every estimator.  This follows the position
:func:`tests.studies.evidence.manifest.study_module_hashes` already takes for a study manifest,
and for the same reason: the record must identify the run well enough to reproduce it, and
re-execution rather than a hash comparison is what keeps the stored numbers honest.
``python scripts/execute_notebook.py <path> --check`` is that re-execution.  It is a command a
contributor runs when a result-determining change lands, in the shape this repository already
uses for a study regeneration, and not a test.

The recorded half also carries :data:`RECORDED_IDENTITY`, which is what a reader acts on when a
digest disagrees.  A digest says two trees differ and never says where, and recomputing one
needs the formula that produced it.  ``cleverly_commit`` names the tree directly, and
``generator_files`` names the set that was folded, so a moved definition reads as a moved
definition rather than as an unexplained hash.

**Coverage boundary.**  The stamp covers the ordered code-cell sources, and each code cell's
identity, execution count, and stored outputs.  It does not cover markdown cells, cell
metadata such as the ``tags`` that ``myst-nb`` reads, or notebook metadata such as
``kernelspec``.  None of those change what the estimators computed, and gating them would make
a typo fix in a markdown cell demand a re-execution.
:mod:`tests.unit.test_documentation_prose` reads the markdown cells separately.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterable
from operator import itemgetter
from pathlib import Path
from typing import Any

__all__ = [
    "GATED_DIGESTS",
    "GENERATOR_MODULES",
    "RECORDED_DIGESTS",
    "RECORDED_IDENTITY",
    "REPOSITORY_ROOT",
    "STAMP_SCHEMA_VERSION",
    "code_cells",
    "code_source_digest",
    "execution_payload_digest",
    "notebook_execution_stamp",
]

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

#: Bumped to 3 when the flat stamp split into a gated half and a recorded half, and to 4 when
#: the recorded half gained the coordinates in :data:`RECORDED_IDENTITY`.
STAMP_SCHEMA_VERSION = 4

#: Asserted equal by the fast tier.  Each one is a digest of the notebook alone.
GATED_DIGESTS = frozenset({"code_source_sha256", "execution_payload_sha256"})

#: Asserted present and well formed by the fast tier, and never asserted equal.
RECORDED_DIGESTS = frozenset(
    {"dependency_lock_sha256", "generator_sha256", "package_source_sha256"}
)

#: Every module that writes a stamp.  The digest helpers moved out of the script, so hashing
#: the script alone would name half of the code that produced the run.
GENERATOR_MODULES = ("scripts/execute_notebook.py", "tests/notebooks.py")

#: What the recorded half carries beside its digests.  A digest is a fold: it says two trees
#: differ and never says where, and it can only be recomputed by whoever still has the formula
#: that produced it.  These four are coordinates a reader acts on directly.
#:
#: :data:`RECORDED_IDENTITY` is the answer to a failure this repository has already had.  The
#: stamp's first ``generator_sha256`` was a plain one-file digest.  The generator later became
#: two files, so the stored value matched no checkout in history, and nothing saw it, because a
#: stranded digest and a current one are the same 64 characters.  ``generator_files`` records
#: the set that was folded, so the fast tier compares definitions rather than guessing.
#:
#: The reasoning is
#: :mod:`tests.studies.evidence.manifest`'s: writing "working tree" records nothing, and what a
#: record must add is the identification a reader needs to reproduce a run.
RECORDED_IDENTITY = frozenset(
    {"cleverly_commit", "cleverly_version", "cleverly_worktree_clean", "generator_files"}
)

#: What :func:`_git` returns when it cannot answer.  A tarball has no repository and a scratch
#: tree has no history, and neither is a reason to fail a stamp.
UNKNOWN = "unknown"


def _git(*arguments: str, root: Path) -> str:
    """Run one git command in ``root``, or return :data:`UNKNOWN` when it cannot run.

    :mod:`tests.studies.evidence.manifest` holds the same three lines.  Importing them here
    would pull that module's registry, and with it 1706 modules including ``numpy``,
    ``pandas``, ``scipy`` and ``sklearn``.  This module is on the fast tier's import path and
    on :mod:`scripts.execute_notebook`'s, so it stays free of the scientific stack.
    """
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN
    return completed.stdout.strip()


def _sha256(payload: bytes) -> str:
    """Return the hexadecimal SHA-256 digest of ``payload``."""
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_digest(value: Any) -> str:
    """Digest JSON data independently of mapping insertion order and display whitespace."""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return _sha256(payload)


def _joined(value: Any) -> Any:
    """Rejoin a split multiline string, and pass anything else through."""
    if isinstance(value, list) and all(isinstance(line, str) for line in value):
        return "".join(value)
    return value


def _is_json_mime(mime: str) -> bool:
    """Report whether a mimebundle key holds JSON data rather than a multiline string."""
    return mime == "application/json" or (
        mime.startswith("application/") and mime.endswith("+json")
    )


def _normalized_output(output: Any) -> dict[str, Any]:
    """Return one stored output with its multiline string fields rejoined.

    ``nbformat`` writes a long string as a list of lines and rejoins it on read, so the same
    output digests two ways depending on whether the caller used :func:`nbformat.read` or
    :func:`json.load`.  This applies the rejoining rule directly, which is the normalization
    chosen for :func:`execution_payload_digest`: the digest is then a property of the stored
    notebook rather than of the reader.  The rule mirrors ``nbformat.v4.rwbase.rejoin_lines``.
    An ``application/json`` payload and an error ``traceback`` are genuine lists and stay
    untouched.
    """
    normalized = dict(output)
    data = normalized.get("data")
    if isinstance(data, dict):
        normalized["data"] = {
            mime: payload if _is_json_mime(mime) else _joined(payload)
            for mime, payload in data.items()
        }
    if "text" in normalized:
        normalized["text"] = _joined(normalized["text"])
    return normalized


def _file_digest(path: Path) -> str:
    """Digest one repository input byte for byte."""
    return _sha256(path.read_bytes())


def _file_set_digest(paths: Iterable[Path], *, root: Path) -> str:
    """Digest a file set, including each repository-relative path and its bytes.

    The sort runs over the built records rather than over the :class:`~pathlib.Path` inputs.
    ``Path.__lt__`` case-folds on Windows and does not on POSIX, so sorting the inputs would
    let one uppercase module name give a Windows digest that Linux CI cannot reproduce.
    Sorting on the ``"path"`` string that is actually hashed removes that dependence.
    """
    resolved_root = root.resolve()
    records = [
        {
            "path": path.resolve().relative_to(resolved_root).as_posix(),
            "sha256": _file_digest(path),
        }
        for path in paths
    ]
    return _canonical_json_digest(sorted(records, key=itemgetter("path")))


def _package_source_digest(repository_root: Path) -> str:
    """Digest every shipped source file below ``src/cleverly``."""
    package_root = repository_root / "src" / "cleverly"
    sources = (
        path
        for path in package_root.rglob("*")
        if path.is_file() and (path.suffix == ".py" or path.name == "py.typed")
    )
    return _file_set_digest(sources, root=repository_root)


def code_cells(notebook: Any) -> list[Any]:
    """Return the code cells of ``notebook`` in document order."""
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def code_source_digest(notebook: Any) -> str:
    """Return a stable digest of the ordered code-cell sources."""
    sources = [_joined(cell["source"]) for cell in code_cells(notebook)]
    return _canonical_json_digest(sources)


def execution_payload_digest(notebook: Any) -> str:
    """Digest the ordered identities, counts, and stored outputs of every code cell.

    Stored outputs pass through :func:`_normalized_output` first, so a notebook read with
    :func:`nbformat.read` and the same notebook read with :func:`json.load` digest alike.
    """
    execution = [
        {
            "id": cell.get("id"),
            "execution_count": cell.get("execution_count"),
            "outputs": [_normalized_output(output) for output in cell.get("outputs", [])],
        }
        for cell in code_cells(notebook)
    ]
    return _canonical_json_digest(execution)


def _recorded_identity(repository_root: Path) -> dict[str, Any]:
    """Name the checkout that ran the notebook, in terms a reader can act on.

    ``cleverly_worktree_clean`` is the honest qualifier on ``cleverly_commit``.  A notebook is
    executed before the commit that lands it, so the recorded commit is the parent and the tree
    was usually dirty.  ``False`` says the commit places the run rather than reproduces it.

    ``cleverly_version`` names the installed package, and ``package_source_sha256`` names the
    tree below ``repository_root``.  The editable install this repository uses makes them the
    same package.  A scratch root separates them, and only the digest follows the root.
    """
    import cleverly

    commit = _git("rev-parse", "HEAD", root=repository_root)
    status = _git("status", "--porcelain", root=repository_root)
    return {
        "cleverly_commit": commit,
        "cleverly_version": cleverly.__version__,
        "cleverly_worktree_clean": status == "" if status != UNKNOWN else None,
        "generator_files": list(GENERATOR_MODULES),
    }


def notebook_execution_stamp(
    notebook: Any,
    notebook_path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Build the two-part reproducibility stamp for one executed notebook."""
    relative = notebook_path.resolve().relative_to(repository_root.resolve()).as_posix()
    return {
        "schema_version": STAMP_SCHEMA_VERSION,
        "command": f"python scripts/execute_notebook.py {relative}",
        "gated": {
            "code_source_sha256": code_source_digest(notebook),
            "execution_payload_sha256": execution_payload_digest(notebook),
        },
        "recorded": {
            "dependency_lock_sha256": _file_digest(repository_root / "uv.lock"),
            "generator_sha256": _file_set_digest(
                (repository_root / module for module in GENERATOR_MODULES),
                root=repository_root,
            ),
            "package_source_sha256": _package_source_digest(repository_root),
            **_recorded_identity(repository_root),
        },
    }
