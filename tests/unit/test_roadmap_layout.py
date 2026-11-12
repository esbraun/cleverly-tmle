"""The roadmap lists each open item once, in the table that orders it.

``docs/roadmap.md`` keeps one contract section per open item and one table row per item: a main
roadmap item in a topic table under "Main roadmap", a future investigation in the "Future
investigations" table. A section with no row is work nobody ranks; a row with no section points at
nothing. A delivered item leaves both, and ``docs/roadmap-delivered.md`` says where its record is.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROADMAP = ROOT / "docs" / "roadmap.md"

#: A contract heading: ``### X91. Survival targets ...``.
CONTRACT = re.compile(r"^### ([A-Z]+\d+[a-z]?)\. ", re.MULTILINE)
#: The item link that opens a table row: ``| [X91](#x91-...)``, with its anchor.
ROW = re.compile(r"^\| \[([A-Z]+\d+[a-z]?)\]\(#([^)]*)\)", re.MULTILINE)
#: The item link in the last cell of a future-investigations row.
FUTURE_ROW = re.compile(r"\| \[([A-Z]+\d+)\]\(#[^)]*\) \|$", re.MULTILINE)


def section(text: str, heading: str) -> str:
    """The body of the level-2 section ``heading``, up to the next level-2 heading."""
    start = text.index(f"\n## {heading}\n")
    end = text.find("\n## ", start + 1)
    return text[start : end if end != -1 else len(text)]


def test_every_main_item_has_one_row_and_one_contract() -> None:
    text = ROADMAP.read_text(encoding="utf-8")
    contracts = CONTRACT.findall(section(text, "Detailed implementation contracts"))
    links = ROW.findall(section(text, "Main roadmap"))
    for item, anchor in links:
        assert anchor.startswith(item.lower() + "-"), (
            f"{item} links another item's contract: #{anchor}"
        )
    rows = [item for item, _ in links]
    assert len(rows) == len(set(rows)), (
        f"items listed twice: {sorted({r for r in rows if rows.count(r) > 1})}"
    )
    assert set(contracts) == set(rows), (
        f"contract with no row: {sorted(set(contracts) - set(rows))}; "
        f"row with no contract: {sorted(set(rows) - set(contracts))}"
    )


def test_every_future_investigation_has_one_row_and_one_contract() -> None:
    text = ROADMAP.read_text(encoding="utf-8")
    contracts = CONTRACT.findall(section(text, "Future investigation contracts"))
    rows = FUTURE_ROW.findall(section(text, "Future investigations"))
    assert len(rows) == len(set(rows))
    assert set(contracts) == set(rows), (
        f"contract with no row: {sorted(set(contracts) - set(rows))}; "
        f"row with no contract: {sorted(set(rows) - set(contracts))}"
    )


def test_every_main_row_names_a_tier() -> None:
    tiers = {"fix", "2", "3", "4", "5", "6"}
    main = section(ROADMAP.read_text(encoding="utf-8"), "Main roadmap")
    for line in main.splitlines():
        if ROW.match(line):
            tier = line.split("|")[2].strip()
            assert tier, f"no tier in: {line[:80]}"
            assert tier.split()[0].rstrip(",;") in tiers, f"no tier in: {line[:80]}"


def test_the_roadmap_names_no_release_milestone() -> None:
    text = ROADMAP.read_text(encoding="utf-8")
    found = re.findall(
        r"\b(?:beta releases?|first beta|after beta|post-beta|milestones?)\b", text, re.I
    )
    assert not found, f"the roadmap orders work by tier, not by release: {found}"


def test_every_fix_owner_is_a_red_cell_owner_and_every_fix_tier_item_is_queued() -> None:
    text = ROADMAP.read_text(encoding="utf-8")
    fixes = section(text, "Fixes")
    owners = set(re.findall(r"^\| `([^`]+)` \|", section(text, "Red-cell owners"), re.MULTILINE))
    queue = [line for line in fixes.splitlines() if re.match(r"^\| \d+ \|", line)]
    named = {name for line in queue for name in re.findall(r"`([^`]+)`", line.split("|")[3])}
    assert named <= owners, f"fix owners missing from Red-cell owners: {sorted(named - owners)}"
    queued_items = {item for line in queue for item in re.findall(r"\[([A-Z]+\d+)\]", line)}
    fix_tier = {
        m.group(1)
        for line in section(text, "Main roadmap").splitlines()
        if (m := ROW.match(line)) and line.split("|")[2].strip().startswith("fix")
    }
    assert fix_tier == queued_items, (
        f"fix-tier items not queued: {sorted(fix_tier - queued_items)}; "
        f"queued items without a fix tier: {sorted(queued_items - fix_tier)}"
    )
