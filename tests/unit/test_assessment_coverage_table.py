"""The assessment coverage table on ``validation-methods.md`` matches the declarations.

The table under ``### Which operations each result family runs`` is rebuilt here from
:data:`cleverly.assessment.ASSESSMENT_CAPABILITIES`, one column per result family and one row
per operation, and compared cell by cell with the page.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from cleverly.assessment import ASSESSMENT_CAPABILITIES, AssessmentCapability, AssessmentStatus

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "technical-reference" / "validation-methods.md"
HEADING = "### Which operations each result family runs"


def _ordered(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _families() -> list[str]:
    return _ordered([row.result_family for row in ASSESSMENT_CAPABILITIES])


def _operations() -> list[str]:
    return _ordered([row.operation for row in ASSESSMENT_CAPABILITIES])


def _family_methods(family: str) -> tuple[str, ...]:
    rows = [row for row in ASSESSMENT_CAPABILITIES if row.result_family == family]
    return tuple(_ordered([method for row in rows for method in row.methods]))


def _cell(row: AssessmentCapability | None, family: str) -> str:
    if row is None:
        return "not declared"
    if not row.available:
        return row.status.value.replace("_", " ")
    text = "runs"
    if row.requires_arguments:
        text += " with " + ", ".join(f"`{name}=`" for name in row.requires_arguments)
    if set(row.methods) != set(_family_methods(family)):
        text += " for " + ", ".join(f"`{m}`" for m in row.methods)
    return text


def expected_table() -> list[list[str]]:
    """The table's rows, header first, as the declarations give them."""
    families = _families()
    index = {(row.result_family, row.operation): row for row in ASSESSMENT_CAPABILITIES}
    rows = [["operation", *(f"`{family}`" for family in families)]]
    rows.append(["methods", *(", ".join(f"`{m}`" for m in _family_methods(f)) for f in families)])
    for operation in _operations():
        cells = [_cell(index.get((family, operation)), family) for family in families]
        rows.append([f"`{operation}`", *cells])
    return rows


def _page_table() -> list[list[str]]:
    text = PAGE.read_text(encoding="utf-8")
    section = text.split(HEADING + "\n", 1)[1].split("\n#", 1)[0]
    lines = [line for line in section.splitlines() if line.startswith("|")]
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
    return [row for row in rows if not set("".join(row)) <= set("-: ")]


def test_each_declaration_has_one_row() -> None:
    keys = [(row.result_family, row.operation) for row in ASSESSMENT_CAPABILITIES]
    assert len(keys) == len(set(keys))


def test_the_page_table_is_the_declarations() -> None:
    assert _page_table() == expected_table()


def test_a_changed_declaration_changes_the_table() -> None:
    """Mutation control: flipping one row's availability moves exactly one cell."""
    row = next(r for r in ASSESSMENT_CAPABILITIES if r.available and not r.requires_arguments)
    assert _cell(row, row.result_family).startswith("runs")
    flipped = replace(row, available=False, status=AssessmentStatus.UNAVAILABLE)
    assert _cell(flipped, row.result_family) == "unavailable"
