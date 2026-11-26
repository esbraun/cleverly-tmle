"""The ``SurvivalDRTMLE`` refusal table on ``scope-and-refusals.md`` is the module's records.

Every :class:`~cleverly._refusal.Refusal` held at module level in
:mod:`cleverly.longitudinal.drift` has one row under ``### Drift-corrected survival TMLE``:
its name, its reason code, its roadmap item, the first sentence of its condition and its
alternative. The table is rebuilt here and compared row by row with the page.
"""

from __future__ import annotations

import re
from pathlib import Path

import cleverly.longitudinal.drift as drift
from cleverly._refusal import Refusal

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "technical-reference" / "scope-and-refusals.md"
HEADING = "### Drift-corrected survival TMLE"


def _first_sentence(text: str) -> str:
    match = re.match(r"(.+?\.)(\s|$)", text.strip())
    return (match.group(1) if match else text.strip()).replace("|", "\\|")


def _cell(text: str | None) -> str:
    return "" if not text else text.strip().replace("|", "\\|")


def expected_rows() -> list[list[str]]:
    """``[name, code, roadmap, condition, alternative]`` per record, in module order."""
    rows = []
    for name, value in vars(drift).items():
        if isinstance(value, Refusal):
            rows.append(
                [
                    f"`{name}`",
                    f"`{value.reason.value}`",
                    _cell(value.roadmap),
                    _first_sentence(value.condition),
                    _cell(value.alternative),
                ]
            )
    return rows


def _page_rows() -> list[list[str]]:
    text = PAGE.read_text(encoding="utf-8")
    section = text.split(HEADING + "\n", 1)[1].split("\n#", 1)[0]
    lines = [line for line in section.splitlines() if line.startswith("| `")]
    return [[cell.strip() for cell in re.split(r"(?<!\\)\|", line.strip())[1:-1]] for line in lines]


def test_the_module_holds_the_records() -> None:
    assert len(expected_rows()) >= 20


def test_the_page_table_is_the_records() -> None:
    assert _page_rows() == expected_rows()
