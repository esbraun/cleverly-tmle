"""The tuned-cap composition table on ``scope-and-refusals.md`` matches the code.

Each ``_composition(...)`` call in ``refuse_tuned_compositions`` names a composition, a reason
code and, for a missing capability, a roadmap item. The table under ``### Data-tuned caps``
must hold a row whose first cell is that composition, whose code cell names that code and
whose roadmap cell names that item.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "technical-reference" / "scope-and-refusals.md"
SOURCE = ROOT / "src" / "cleverly" / "estimators" / "_tuned_policies.py"


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next((k.value for k in call.keywords if k.arg == name), None)


def code_compositions() -> list[tuple[str, str, str | None]]:
    """``(composition, code, roadmap)`` of every ``_composition`` call in the function."""
    module = ast.parse(SOURCE.read_text(encoding="utf-8"))
    function = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == "refuse_tuned_compositions"
    )
    found = []
    for node in ast.walk(function):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_composition"
        ):
            continue
        what = node.args[0]
        code = _keyword(node, "code")
        roadmap = _keyword(node, "roadmap")
        assert isinstance(what, ast.Constant) and isinstance(code, ast.Attribute)
        found.append(
            (
                str(what.value),
                code.attr.lower(),
                None if roadmap is None else str(ast.literal_eval(roadmap)),
            )
        )
    return found


def page_rows() -> dict[str, list[str]]:
    """The composition table's rows, keyed by the first cell with code spans unwrapped."""
    text = PAGE.read_text(encoding="utf-8")
    section = text.split("### Data-tuned caps\n", 1)[1].split("\n#", 1)[0]
    table = section.split("| composition |", 1)[1].split("\n\n", 1)[0]
    rows = {}
    for line in table.splitlines()[2:]:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows[cells[0].replace("`", "")] = cells
    return rows


def test_the_scan_finds_the_compositions() -> None:
    assert len(code_compositions()) >= 8


def test_each_composition_has_a_row_with_its_code_and_roadmap() -> None:
    rows = page_rows()
    for what, code, roadmap in code_compositions():
        assert what in rows, what
        assert f"`{code}`" in rows[what][1], (what, code)
        if roadmap is None:
            assert rows[what][2] == "", what
        else:
            assert f"[{roadmap}]" in rows[what][2], (what, roadmap)


def test_the_table_has_no_row_the_code_lacks() -> None:
    assert set(page_rows()) == {what for what, _, _ in code_compositions()}
