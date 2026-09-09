"""Execute and provenance-stamp a committed notebook artifact.

Running this needs a network connection, a ``python3`` kernel, and the time it takes to refit
every estimator the notebook fits. Documentation builds stay offline and render the committed
outputs.

The stamp itself is built by :mod:`tests.notebooks`, which also says which half of it the fast
tier asserts equal. Keeping the digests there keeps :mod:`nbclient` out of every fast-tier run:
verification needs the digests, and only this entry point needs a kernel.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    """Execute the notebook named on the command line, then write its stamp back.

    ``python scripts/execute_notebook.py <path>`` puts ``scripts/`` on the import path and not
    the repository root, so the stamp module is importable only after the first line here.
    That command is the one the stamp records and the one the contributor documentation gives,
    so the path repair belongs in the script rather than in an instruction to export
    ``PYTHONPATH``.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tests.notebooks import REPOSITORY_ROOT, notebook_execution_stamp

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
