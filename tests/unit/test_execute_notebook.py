"""Mutation controls for the shared notebook artifact provenance stamp.

Every key the stamp carries needs its own witness, because a digest that stops covering an
input still returns a hash and still compares equal to itself.  The witnesses run against a
scratch repository rather than the real one, so a control here costs no re-execution.
"""

from __future__ import annotations

import copy
import json
import re
from operator import itemgetter
from pathlib import Path
from typing import Any

import pytest

from scripts.execute_notebook import WALL_CLOCK, text_outputs
from tests.notebooks import (
    GATED_DIGESTS,
    GENERATOR_MODULES,
    RECORDED_DIGESTS,
    RECORDED_IDENTITY,
    REPOSITORY_ROOT,
    STAMP_SCHEMA_VERSION,
    UNKNOWN,
    _canonical_json_digest,
    _file_digest,
    _file_set_digest,
    code_source_digest,
    execution_payload_digest,
    notebook_execution_stamp,
)


def _repository(root: Path) -> Path:
    """Build the repository inputs the stamp expects beneath ``root``, and return the notebook.

    The notebook file only has to exist, because :func:`notebook_execution_stamp` digests the
    payload it is handed and reads the path for the repair command it records.
    """
    for module in GENERATOR_MODULES:
        generator = root / module
        generator.parent.mkdir(parents=True, exist_ok=True)
        generator.write_text(f"# {module}\n", encoding="utf-8")
    (root / "src" / "cleverly").mkdir(parents=True)
    (root / "src" / "cleverly" / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "src" / "cleverly" / "py.typed").write_text("", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    notebook = root / "docs" / "example.ipynb"
    notebook.parent.mkdir()
    notebook.write_text(json.dumps(_notebook()), encoding="utf-8")
    return notebook


def _payload_notebook(mime: str, payload: Any) -> dict[str, Any]:
    """Return a notebook whose one code cell carries ``payload`` under ``mime``."""
    return {
        "cells": [
            {
                "cell_type": "code",
                "id": "answer",
                "source": "render()",
                "execution_count": 1,
                "outputs": [{"output_type": "display_data", "data": {mime: payload}}],
            }
        ]
    }


def _notebook() -> dict[str, Any]:
    """Return one minimal executed notebook payload."""
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "prose",
                "source": "# Title\n",
            },
            {
                "cell_type": "code",
                "id": "answer",
                "source": "print(1)",
                "execution_count": 1,
                "outputs": [{"output_type": "stream", "name": "stdout", "text": "1\n"}],
            },
        ]
    }


#: Each digest the stamp carries, and the test here that fails when its contribution is removed.
#: Written out rather than derived from :data:`GATED_DIGESTS` and :data:`RECORDED_DIGESTS`.  A
#: map built from the constant it checks moves whenever that constant moves, so the two terms
#: cancel and the assertion holds for a field nobody witnessed.
WITNESSED_BY = {
    "code_source_sha256": "test_each_code_cell_field_moves_exactly_one_gated_digest",
    "execution_payload_sha256": "test_each_code_cell_field_moves_exactly_one_gated_digest",
    "generator_sha256": "test_repository_inputs_each_invalidate_the_stamp",
    "package_source_sha256": "test_repository_inputs_each_invalidate_the_stamp",
    "dependency_lock_sha256": "test_repository_inputs_each_invalidate_the_stamp",
    "cleverly_commit": "test_the_recorded_identity_names_the_checkout",
    "cleverly_version": "test_the_recorded_identity_names_the_checkout",
    "cleverly_worktree_clean": "test_the_recorded_identity_names_the_checkout",
    "generator_files": "test_the_generator_file_set_is_recorded_beside_its_digest",
}


def test_the_stamp_carries_exactly_the_keys_this_module_witnesses(tmp_path: Path) -> None:
    """A new stamp field is unwitnessed until somebody adds its control, so name them here.

    This assertion is the reason the rest of the module can be trusted.  Without it a field
    added to the stamp would be covered by nothing, and the suite would still be green.  Every
    set below is a literal, so adding a digest fails here until its witness is named.
    """
    path = _repository(tmp_path)
    stamp = notebook_execution_stamp(_notebook(), path, repository_root=tmp_path)

    assert set(stamp) == {"schema_version", "command", "gated", "recorded"}, (
        "the stamp gained or lost a top-level key; give it a witness in this module before you "
        "change this assertion"
    )
    assert set(stamp["gated"]) == {"code_source_sha256", "execution_payload_sha256"}, (
        "the gated half changed; add the new digest to WITNESSED_BY with the control that "
        "fails when its contribution is removed"
    )
    assert set(stamp["recorded"]) == {
        "generator_sha256",
        "package_source_sha256",
        "dependency_lock_sha256",
        "cleverly_commit",
        "cleverly_version",
        "cleverly_worktree_clean",
        "generator_files",
    }, "the recorded half changed; add the new field to WITNESSED_BY with its control"

    assert set(WITNESSED_BY) == set(stamp["gated"]) | set(stamp["recorded"])
    assert set(stamp["gated"]) == set(GATED_DIGESTS)
    assert set(stamp["recorded"]) == set(RECORDED_DIGESTS) | set(RECORDED_IDENTITY)

    missing = sorted(name for name in set(WITNESSED_BY.values()) if name not in globals())
    assert not missing, f"WITNESSED_BY names test(s) this module does not define: {missing}"


def test_the_generator_file_set_is_recorded_beside_its_digest(tmp_path: Path) -> None:
    """The stamp names the files it folded, so a moved definition is readable.

    A digest is only recomputable by whoever still has the formula behind it.  The first
    ``generator_sha256`` folded one file, the generator became two, and the stored value then
    matched no checkout in history with nothing to say so.  The recorded set is what the fast
    tier compares, so the next definition change reports itself.
    """
    path = _repository(tmp_path)
    stamp = notebook_execution_stamp(_notebook(), path, repository_root=tmp_path)

    assert stamp["recorded"]["generator_files"] == list(GENERATOR_MODULES)
    assert len(stamp["recorded"]["generator_files"]) > 1, (
        "one entry would not separate a recorded set from a recorded path, which is the "
        "distinction this field exists to make"
    )

    folded = _file_set_digest(
        (tmp_path / module for module in stamp["recorded"]["generator_files"]),
        root=tmp_path,
    )
    assert folded == stamp["recorded"]["generator_sha256"], (
        "the recorded file set does not reproduce the recorded digest, so the two describe "
        "different runs"
    )


def test_the_recorded_identity_names_the_checkout(tmp_path: Path) -> None:
    """A commit, a version, and an honest qualifier on the commit.

    ``cleverly_worktree_clean`` is what keeps ``cleverly_commit`` from overclaiming.  A notebook
    is executed before the commit that lands it, so the recorded commit is the parent.  A
    scratch root is no repository at all, and the stamp reports that rather than failing.
    """
    path = _repository(tmp_path)
    scratch = notebook_execution_stamp(_notebook(), path, repository_root=tmp_path)

    assert scratch["recorded"]["cleverly_commit"] == UNKNOWN
    assert scratch["recorded"]["cleverly_worktree_clean"] is None
    assert isinstance(scratch["recorded"]["cleverly_version"], str)
    assert scratch["recorded"]["cleverly_version"]

    real = notebook_execution_stamp(_notebook(), REPOSITORY_ROOT / "docs" / "x.ipynb")
    assert re.fullmatch(r"[0-9a-f]{40}", real["recorded"]["cleverly_commit"]), (
        "this repository has a history, so the stamp has to name a revision here"
    )
    assert isinstance(real["recorded"]["cleverly_worktree_clean"], bool)


def test_the_schema_version_and_command_are_part_of_the_stamp(tmp_path: Path) -> None:
    """Both identify what to compare and how to repair it, so both are asserted equal."""
    path = _repository(tmp_path)
    stamp = notebook_execution_stamp(_notebook(), path, repository_root=tmp_path)

    assert stamp["schema_version"] == STAMP_SCHEMA_VERSION
    assert stamp["command"] == "python scripts/execute_notebook.py docs/example.ipynb"

    moved = tmp_path / "docs" / "renamed.ipynb"
    moved.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    relocated = notebook_execution_stamp(_notebook(), moved, repository_root=tmp_path)

    assert relocated["command"] != stamp["command"]


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("source", lambda cell: cell.__setitem__("source", "print(2)")),
        ("id", lambda cell: cell.__setitem__("id", "renamed")),
        ("execution_count", lambda cell: cell.__setitem__("execution_count", 2)),
        ("outputs", lambda cell: cell["outputs"][0].__setitem__("text", "2\n")),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_each_code_cell_field_moves_exactly_one_gated_digest(
    field: str, mutate: Any, tmp_path: Path
) -> None:
    """``source`` moves the code digest; identity, count, and outputs move the payload digest.

    Splitting the two is what makes a failure readable.  A single digest over both would say
    the notebook is stale and leave the reader to work out which half.
    """
    path = _repository(tmp_path)
    notebook = _notebook()
    original = notebook_execution_stamp(notebook, path, repository_root=tmp_path)

    changed = copy.deepcopy(notebook)
    mutate(changed["cells"][1])
    current = notebook_execution_stamp(changed, path, repository_root=tmp_path)

    moved = {key for key in GATED_DIGESTS if current["gated"][key] != original["gated"][key]}
    expected = {"code_source_sha256"} if field == "source" else {"execution_payload_sha256"}

    assert moved == expected


def test_a_markdown_cell_is_outside_the_stamp(tmp_path: Path) -> None:
    """The documented coverage boundary, asserted rather than described.

    ``tests/notebooks.py`` says the stamp reaches code cells only, and this is what makes that
    sentence checkable.  A markdown edit needs no re-execution, so gating it would demand one
    for a typo fix.
    """
    path = _repository(tmp_path)
    notebook = _notebook()
    original = notebook_execution_stamp(notebook, path, repository_root=tmp_path)

    changed = copy.deepcopy(notebook)
    changed["cells"][0]["source"] = "# Retitled\n"
    current = notebook_execution_stamp(changed, path, repository_root=tmp_path)

    assert current["gated"] == original["gated"]


def test_a_split_multiline_output_digests_like_the_joined_one() -> None:
    """A notebook read as plain JSON digests like the same notebook read through ``nbformat``.

    ``nbformat`` writes a long string as a list of lines and rejoins it on read, so without an
    explicit normalization the same file gives two payload digests depending on the reader.
    The measured pair on the TWINS artifact was ``f148d7f3`` in memory and ``f50275c2`` from
    :func:`json.load`.
    """
    joined = _notebook()
    joined["cells"][1]["outputs"][0]["text"] = "one\ntwo\n"
    split = copy.deepcopy(joined)
    split["cells"][1]["outputs"][0]["text"] = ["one\n", "two\n"]
    split["cells"][1]["source"] = ["print", "(1)"]
    joined["cells"][1]["source"] = "print(1)"

    assert execution_payload_digest(split) == execution_payload_digest(joined)
    assert code_source_digest(split) == code_source_digest(joined)


def test_the_normalization_still_separates_two_different_outputs() -> None:
    """The control on the control: a join that flattened everything would pass the test above.

    Two line lists that join to different strings must keep different digests, and an error
    ``traceback`` is a genuine list that no rejoining rule may collapse.
    """
    left = _notebook()
    left["cells"][1]["outputs"][0]["text"] = ["one\n", "two\n"]
    right = copy.deepcopy(left)
    right["cells"][1]["outputs"][0]["text"] = ["one\n", "three\n"]

    assert execution_payload_digest(left) != execution_payload_digest(right)

    traced = _notebook()
    traced["cells"][1]["outputs"] = [
        {"output_type": "error", "ename": "E", "evalue": "v", "traceback": ["a", "b"]}
    ]
    reordered = copy.deepcopy(traced)
    reordered["cells"][1]["outputs"][0]["traceback"] = ["b", "a"]

    assert execution_payload_digest(traced) != execution_payload_digest(reordered)


def test_a_json_payload_keeps_its_list_structure() -> None:
    """A JSON mimebundle is data, so joining its lines would merge two different payloads.

    ``nbformat`` splits a multiline string and leaves a JSON payload alone, and
    :func:`tests.notebooks._normalized_output` has to make the same distinction.  Treating
    ``application/json`` as text would join both lists below to ``"ab"`` and give them one
    digest, which is the collision this control exists to see.
    """
    for mime in ("application/json", "application/vnd.plotly.v1+json"):
        split = _payload_notebook(mime, ["a", "b"])
        joined = _payload_notebook(mime, ["ab"])
        assert execution_payload_digest(split) != execution_payload_digest(joined), mime

    text = _payload_notebook("text/plain", ["a", "b"])
    merged = _payload_notebook("text/plain", ["ab"])
    assert execution_payload_digest(text) == execution_payload_digest(merged)


def test_the_package_digest_covers_py_typed_and_nothing_generated(tmp_path: Path) -> None:
    """``py.typed`` ships and moves the digest.  A build artifact does not ship and does not.

    The filter has to admit one non-``.py`` file and refuse the rest.  Without the second half
    a widened filter would pull ``__pycache__`` in, and the digest would then depend on which
    interpreters had imported the package.
    """
    path = _repository(tmp_path)
    notebook = _notebook()
    package = tmp_path / "src" / "cleverly"
    original = notebook_execution_stamp(notebook, path, repository_root=tmp_path)

    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "__init__.cpython-313.pyc").write_bytes(b"\x00compiled")
    (package / "notes.md").write_text("not shipped as source\n", encoding="utf-8")
    unchanged = notebook_execution_stamp(notebook, path, repository_root=tmp_path)
    assert unchanged["recorded"] == original["recorded"]

    (package / "py.typed").unlink()
    without = notebook_execution_stamp(notebook, path, repository_root=tmp_path)
    assert (
        without["recorded"]["package_source_sha256"]
        != (original["recorded"]["package_source_sha256"])
    )


@pytest.mark.parametrize(
    ("field", "relative"),
    [
        ("generator_sha256", "scripts/execute_notebook.py"),
        ("generator_sha256", "tests/notebooks.py"),
        ("package_source_sha256", "src/cleverly/__init__.py"),
        ("dependency_lock_sha256", "uv.lock"),
    ],
    ids=lambda value: value,
)
def test_repository_inputs_each_invalidate_the_stamp(
    field: str, relative: str, tmp_path: Path
) -> None:
    """Each recorded input is one independent result, so a failure names the input it lost.

    The loop this replaced aborted on its first failure and printed two hashes that read
    alike.  Recorded is not ungated: the fast tier does not assert these equal, and a stamp
    that could not see them would identify no checkout at all.
    """
    path = _repository(tmp_path)
    notebook = _notebook()
    original = notebook_execution_stamp(notebook, path, repository_root=tmp_path)

    changed = tmp_path / relative
    changed.write_text(changed.read_text(encoding="utf-8") + "changed\n", encoding="utf-8")
    current = notebook_execution_stamp(notebook, path, repository_root=tmp_path)

    assert current["recorded"][field] != original["recorded"][field]
    assert current["gated"] == original["gated"]


def test_package_file_membership_is_part_of_the_source_digest(tmp_path: Path) -> None:
    """Both an added and a deleted shipped file move the digest, with every other byte fixed."""
    path = _repository(tmp_path)
    notebook = _notebook()
    added = tmp_path / "src" / "cleverly" / "added.py"
    original = notebook_execution_stamp(notebook, path, repository_root=tmp_path)

    added.write_text("VALUE = 2\n", encoding="utf-8")
    grown = notebook_execution_stamp(notebook, path, repository_root=tmp_path)
    added.unlink()
    (tmp_path / "src" / "cleverly" / "__init__.py").unlink()
    shrunk = notebook_execution_stamp(notebook, path, repository_root=tmp_path)

    assert (
        grown["recorded"]["package_source_sha256"] != original["recorded"]["package_source_sha256"]
    )
    assert (
        shrunk["recorded"]["package_source_sha256"] != original["recorded"]["package_source_sha256"]
    )
    assert shrunk["recorded"]["package_source_sha256"] != grown["recorded"]["package_source_sha256"]


def test_a_file_set_digest_is_independent_of_input_order(tmp_path: Path) -> None:
    """Order in, one digest out.  ``rglob`` gives no order the two platforms agree on."""
    for name in ("Alpha.py", "beta.py", "Gamma.py"):
        (tmp_path / name).write_text(f"# {name}\n", encoding="utf-8")
    paths = [tmp_path / name for name in ("Alpha.py", "beta.py", "Gamma.py")]

    digests = {
        _file_set_digest(order, root=tmp_path)
        for order in (paths, list(reversed(paths)), [paths[1], paths[2], paths[0]])
    }

    assert len(digests) == 1


def test_a_file_set_digest_sorts_on_the_string_it_hashes(tmp_path: Path) -> None:
    """One uppercase module name must not give Windows and Linux two different digests.

    ``Path.__lt__`` case-folds on Windows and does not on POSIX, so sorting the inputs made
    the digest a property of the operating system.  ``"Beta.py"`` sorts before ``"alpha.py"``
    by code point and after it when folded, so the two orders here are genuinely different.
    The digest must be the one the code points give.
    """
    names = ("Beta.py", "alpha.py")
    for name in names:
        (tmp_path / name).write_text(f"# {name}\n", encoding="utf-8")
    records = [{"path": name, "sha256": _file_digest(tmp_path / name)} for name in names]
    by_code_point = _canonical_json_digest(sorted(records, key=itemgetter("path")))
    when_folded = _canonical_json_digest(sorted(records, key=lambda r: r["path"].casefold()))

    assert by_code_point != when_folded, "the witness needs two orders that actually differ"
    assert _file_set_digest([tmp_path / name for name in names], root=tmp_path) == by_code_point


def test_a_file_set_digest_moves_when_a_file_is_renamed(tmp_path: Path) -> None:
    """The witness the path key had none of: identical bytes under two names.

    Dropping ``"path"`` from the record leaves every digest in this module unchanged, because
    the bytes are what the other controls move.  A rename is the only mutation that separates
    the two, and it is also what a case-folded sort gets wrong.
    """
    left = tmp_path / "alpha.py"
    right = tmp_path / "Beta.py"
    left.write_text("SAME = 1\n", encoding="utf-8")
    right.write_text("SAME = 1\n", encoding="utf-8")
    before = _file_set_digest([left], root=tmp_path)
    after = _file_set_digest([right], root=tmp_path)

    assert before != after


def test_the_wall_clock_mask_hides_only_the_clock() -> None:
    """The mask removes the timestamp and leaves every number an estimate could move.

    ``--check`` compares the text a notebook prints against the text it stores, and one field in
    that text is a wall clock: :func:`cleverly.provenance.describe` writes
    ``datetime.now(UTC).isoformat(timespec="seconds")``.  Without the mask the comparison reports
    a difference on every run and can never pass.  A mask wide enough to hide a changed estimate
    would make it unable to fail, which is the worse failure of the two.
    """
    stamp = "data f58e9392 | folds 404b7cec | cleverly 0.1.0 | 2026-09-08T23:26:45+00:00"
    later = "data f58e9392 | folds 404b7cec | cleverly 0.1.0 | 2026-09-09T01:54:45+00:00"
    assert WALL_CLOCK.sub("<wall clock>", stamp) == WALL_CLOCK.sub("<wall clock>", later)
    assert "2026" not in WALL_CLOCK.sub("<wall clock>", stamp)

    moved = "ate_regimen[always vs never]  -0.0639   0.0128      [-0.0889, -0.0389]"
    assert WALL_CLOCK.sub("<wall clock>", moved) == moved
    assert WALL_CLOCK.sub("<wall clock>", "cleverly 0.1.0") == "cleverly 0.1.0"


def test_the_check_comparison_reads_code_cell_text_and_not_images() -> None:
    """``--check`` compares printed text, and a re-rendered figure is not a moved number."""
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "prose", "source": "# Title\n"},
            {
                "cell_type": "code",
                "id": "answer",
                "source": "render()",
                "outputs": [
                    {"output_type": "stream", "text": "psi = -0.0639\n"},
                    {"output_type": "display_data", "data": {"image/png": "iVBORw0KGgo="}},
                ],
            },
        ]
    }
    collected = text_outputs(notebook)

    assert set(collected) == {"answer"}, "a markdown cell prints nothing and is not compared"
    assert any("-0.0639" in text for text in collected["answer"])
    assert not any("iVBORw0" in text for text in collected["answer"])
