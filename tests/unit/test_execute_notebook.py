"""Mutation controls for the shared notebook artifact provenance stamp."""

from __future__ import annotations

import copy
from pathlib import Path

from scripts.execute_notebook import notebook_execution_stamp


def _repository(root: Path) -> Path:
    """Build the repository inputs the stamp expects beneath ``root``."""
    (root / "scripts").mkdir()
    (root / "scripts" / "execute_notebook.py").write_text("generator\n", encoding="utf-8")
    (root / "src" / "cleverly").mkdir(parents=True)
    (root / "src" / "cleverly" / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    notebook = root / "docs" / "example.ipynb"
    notebook.parent.mkdir()
    notebook.write_text("{}\n", encoding="utf-8")
    return notebook


def _notebook() -> dict[str, object]:
    """Return one minimal executed notebook payload."""
    return {
        "cells": [
            {
                "cell_type": "code",
                "id": "answer",
                "source": "print(1)",
                "execution_count": 1,
                "outputs": [{"output_type": "stream", "name": "stdout", "text": "1\n"}],
            }
        ]
    }


def test_code_and_execution_payloads_have_independent_witnesses(tmp_path: Path) -> None:
    """A changed cell or stored result moves its own digest and leaves the other one stable."""
    path = _repository(tmp_path)
    notebook = _notebook()
    original = notebook_execution_stamp(notebook, path, repository_root=tmp_path)

    changed_source = copy.deepcopy(notebook)
    changed_source["cells"][0]["source"] = "print(2)"  # type: ignore[index]
    source_stamp = notebook_execution_stamp(changed_source, path, repository_root=tmp_path)
    assert source_stamp["code_source_sha256"] != original["code_source_sha256"]
    assert source_stamp["execution_payload_sha256"] == original["execution_payload_sha256"]

    changed_output = copy.deepcopy(notebook)
    changed_output["cells"][0]["outputs"][0]["text"] = "2\n"  # type: ignore[index]
    output_stamp = notebook_execution_stamp(changed_output, path, repository_root=tmp_path)
    assert output_stamp["execution_payload_sha256"] != original["execution_payload_sha256"]
    assert output_stamp["code_source_sha256"] == original["code_source_sha256"]


def test_repository_inputs_each_invalidate_the_stamp(tmp_path: Path) -> None:
    """Generator, shipped source, and dependency changes cannot retain a current stamp."""
    path = _repository(tmp_path)
    notebook = _notebook()
    original = notebook_execution_stamp(notebook, path, repository_root=tmp_path)

    inputs = (
        ("generator_sha256", tmp_path / "scripts" / "execute_notebook.py"),
        ("package_source_sha256", tmp_path / "src" / "cleverly" / "__init__.py"),
        ("dependency_lock_sha256", tmp_path / "uv.lock"),
    )
    for field, changed in inputs:
        prior = changed.read_text(encoding="utf-8")
        changed.write_text(prior + "changed\n", encoding="utf-8")
        current = notebook_execution_stamp(notebook, path, repository_root=tmp_path)
        assert current[field] != original[field]
        changed.write_text(prior, encoding="utf-8")


def test_package_file_membership_is_part_of_the_source_digest(tmp_path: Path) -> None:
    """Adding a shipped package file invalidates a stamp even if existing bytes stay fixed."""
    path = _repository(tmp_path)
    notebook = _notebook()
    original = notebook_execution_stamp(notebook, path, repository_root=tmp_path)

    (tmp_path / "src" / "cleverly" / "added.py").write_text("VALUE = 2\n", encoding="utf-8")
    current = notebook_execution_stamp(notebook, path, repository_root=tmp_path)

    assert current["package_source_sha256"] != original["package_source_sha256"]
