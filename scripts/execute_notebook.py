"""Execute and stamp a committed notebook artifact.

The stamp ties stored outputs to the code-cell sources.  It does not make a notebook a
scientific test; it only prevents an edited code cell from silently retaining outputs from an
older execution.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient


def code_source_digest(notebook: Any) -> str:
    """Return a stable digest of the ordered code-cell sources."""
    sources = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    payload = "\n\n# --- notebook cell ---\n\n".join(sources).encode()
    return hashlib.sha256(payload).hexdigest()


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
        resources={"metadata": {"path": str(Path.cwd())}},
    ).execute()
    notebook.metadata["cleverly_execution"] = {
        "code_source_sha256": code_source_digest(notebook),
        "command": f"python scripts/execute_notebook.py {args.notebook.as_posix()}",
    }
    nbformat.write(notebook, path)


if __name__ == "__main__":
    main()
