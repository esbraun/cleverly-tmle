"""The vector-treatment row that X31 retired leaves no name behind.

Every refusal that once deferred to that row now carries its own disposition: a shipped
composition with a witness, or a refusal with a theoretical reason.  No roadmap row of that
number exists, so a source, document or test that names it, or the constant that held it,
points a reader at nothing.  The names are assembled here so this file does not match
itself.  A notebook is read by its cell sources: its stored outputs hold base64 image data,
which can contain any short string.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RETIRED = ("X" + "37", "VECTOR_DENSITY" + "_ROW")
SKIPPED_DIRECTORIES = {"__pycache__", "_build", ".ipynb_checkpoints"}


def _text_of(path: Path) -> str | None:
    """The searchable text of one file: a notebook's cell sources, or a text file's body."""
    if path.suffix == ".ipynb":
        notebook = json.loads(path.read_text(encoding="utf-8"))
        return "\n".join("".join(cell.get("source", ())) for cell in notebook.get("cells", ()))
    raw = path.read_bytes()
    if b"\0" in raw:
        return None
    return raw.decode("utf-8", errors="ignore")


def test_no_source_document_or_test_names_the_retired_row() -> None:
    found = []
    for top in ("src", "docs", "tests"):
        for path in sorted((ROOT / top).rglob("*")):
            if not path.is_file() or SKIPPED_DIRECTORIES & set(path.parts):
                continue
            text = _text_of(path)
            if text is None:
                continue
            found.extend(f"{path.relative_to(ROOT)}: {name}" for name in RETIRED if name in text)
    assert found == []


def test_the_scan_reads_a_notebook_by_its_cell_sources(tmp_path: Path) -> None:
    """A name in an output is not a name in the notebook; a name in a cell is."""
    notebook = {
        "cells": [{"cell_type": "markdown", "source": ["see ", RETIRED[0]]}],
        "metadata": {"outputs": RETIRED[1]},
    }
    path = tmp_path / "example.ipynb"
    path.write_text(json.dumps(notebook), encoding="utf-8")
    text = _text_of(path)
    assert text is not None and RETIRED[0] in text and RETIRED[1] not in text
