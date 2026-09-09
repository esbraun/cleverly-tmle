"""Execute and provenance-stamp a committed notebook artifact.

Running this needs a network connection, a ``python3`` kernel, and the time it takes to refit
every estimator the notebook fits. Documentation builds stay offline and render the committed
outputs.

The stamp itself is built by :mod:`tests.notebooks`, which also says which half of it the fast
suite asserts equal. Keeping the digests there keeps :mod:`nbclient` out of every fast-suite run:
verification needs the digests, and only this entry point needs a kernel.

``--check`` re-executes without writing and reports each cell whose printed text moved. The fast
suite compares digests, and a digest cannot see a library change that moves a published number
while every cell keeps its bytes. That is not hypothetical: one re-execution moved the TWINS
estimate from ``-0.0651`` to ``-0.0639`` with the code cells untouched. This is the notebook's
form of the rule ``docs/development/method-benchmarking.md`` states for a study, that
re-execution rather than a hash comparison is what keeps a stored number honest.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient

#: The wall clock a fit prints in its provenance stamp. ``cleverly.provenance`` writes
#: ``datetime.now(UTC).isoformat(timespec="seconds")``, so this moves on every re-execution and
#: reports nothing about an estimate. Masked rather than dropped, so a stamp that disappears
#: altogether still reads as a difference.
WALL_CLOCK = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2}|Z)?")


def text_outputs(notebook: Any) -> dict[str, list[str]]:
    """Cell id to the text every output of that cell carries, in order.

    Images are left out. A figure re-renders to different bytes for reasons that have nothing to
    do with an estimate, and this comparison reports a moved number.
    """
    collected: dict[str, list[str]] = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        texts = []
        for output in cell.get("outputs", ()):
            texts.append(WALL_CLOCK.sub("<wall clock>", str(output.get("text", ""))))
            texts.append(
                WALL_CLOCK.sub("<wall clock>", str(output.get("data", {}).get("text/plain", "")))
            )
        collected[cell["id"]] = texts
    return collected


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
    parser.add_argument(
        "--check",
        action="store_true",
        help="re-execute and report moved output text without writing the notebook",
    )
    args = parser.parse_args()

    path = args.notebook.resolve()
    notebook = nbformat.read(path, as_version=4)
    stored = text_outputs(notebook)
    NotebookClient(
        notebook,
        timeout=args.timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPOSITORY_ROOT)}},
    ).execute()

    if args.check:
        fresh = text_outputs(notebook)
        moved = sorted(cell for cell in stored if stored[cell] != fresh.get(cell))
        if moved:
            relative = path.relative_to(REPOSITORY_ROOT).as_posix()
            print(f"{relative} printed different text in {moved}")
            print("Read the difference before you regenerate. The stored artifact is what the")
            print("published prose interprets.")
            raise SystemExit(1)
        print(f"{path.name}: every cell printed the text it stores")
        return

    notebook.metadata["cleverly_execution"] = notebook_execution_stamp(notebook, path)
    nbformat.write(notebook, path)


if __name__ == "__main__":
    main()
