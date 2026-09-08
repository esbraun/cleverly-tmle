"""Execute and provenance-stamp a committed notebook artifact.

The stamp binds stored execution payloads to every input this repository controls: the ordered
code cells, the notebook generator, the shipped ``cleverly`` source tree, and the dependency
lock. Documentation builds stay offline and render the committed outputs. The fast tier checks
the same stamp for every reader-facing notebook.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STAMP_SCHEMA_VERSION = 2


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


def code_source_digest(notebook: Any) -> str:
    """Return a stable digest of the ordered code-cell sources."""
    sources = [
        "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    return _canonical_json_digest(sources)


def execution_payload_digest(notebook: Any) -> str:
    """Digest the ordered identities, counts, and stored outputs of every code cell."""
    execution = [
        {
            "id": cell.get("id"),
            "execution_count": cell.get("execution_count"),
            "outputs": cell.get("outputs", []),
        }
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    return _canonical_json_digest(execution)


def file_digest(path: Path) -> str:
    """Digest one repository input byte for byte."""
    return _sha256(path.read_bytes())


def file_set_digest(paths: Iterable[Path], *, root: Path) -> str:
    """Digest a sorted file set, including each repository-relative path and its bytes."""
    records = [
        {
            "path": path.resolve().relative_to(root.resolve()).as_posix(),
            "sha256": file_digest(path),
        }
        for path in sorted(paths)
    ]
    return _canonical_json_digest(records)


def package_source_digest(repository_root: Path = REPOSITORY_ROOT) -> str:
    """Digest every shipped source file below ``src/cleverly``."""
    package_root = repository_root / "src" / "cleverly"
    sources = (
        path
        for path in package_root.rglob("*")
        if path.is_file() and (path.suffix == ".py" or path.name == "py.typed")
    )
    return file_set_digest(sources, root=repository_root)


def notebook_execution_stamp(
    notebook: Any,
    notebook_path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Build the complete reproducibility stamp for one executed notebook."""
    relative = notebook_path.resolve().relative_to(repository_root.resolve()).as_posix()
    return {
        "schema_version": STAMP_SCHEMA_VERSION,
        "code_source_sha256": code_source_digest(notebook),
        "execution_payload_sha256": execution_payload_digest(notebook),
        "generator_sha256": file_digest(repository_root / "scripts" / "execute_notebook.py"),
        "package_source_sha256": package_source_digest(repository_root),
        "dependency_lock_sha256": file_digest(repository_root / "uv.lock"),
        "command": f"python scripts/execute_notebook.py {relative}",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    path = args.notebook.resolve()
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(
        notebook,
        timeout=args.timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPOSITORY_ROOT)}},
    ).execute()
    notebook.metadata["cleverly_execution"] = notebook_execution_stamp(notebook, path)
    nbformat.write(notebook, path)


if __name__ == "__main__":
    main()
