"""The roadmap lists each open item once, in the table that orders it.

``docs/roadmap.md`` keeps one contract section per open item and one table row per item: a main
roadmap item in a topic table under "Main roadmap", a future investigation in the "Future
investigations" table. A section with no row is work nobody ranks; a row with no section points at
nothing. A delivered item leaves both, and ``docs/roadmap-delivered.md`` says where its record is.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.documents import READER_FACING

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
    tiers = {"docs", "scale", "fix", "2", "3", "4", "5", "6"}
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


# ---------------------------------------------------------------- removed items stay removed
#
# A delivered item leaves the roadmap, and every refusal, docstring and page that named it as
# the owner of a gap must move with it. The generic check reads every place a text names a
# roadmap item as the owner of something and requires the item to have a heading, so a
# future removal needs no hand-kept tuple of removed IDs. The P1-specific checks below catch
# the forms the generic patterns cannot see: a bare ``P1`` on a page, and ``outside P1``.

SRC = ROOT / "src" / "cleverly"
DELIVERED = ROOT / "docs" / "roadmap-delivered.md"

_ID = r"[XFRP]\d+[a-z]?"
#: Every form in which a text names a roadmap item as the owner of something.
OWNER_FORMS = (
    re.compile(rf"\b({_ID}) in (?:``)?(?:docs/roadmap\.md|the roadmap)\b"),
    re.compile(rf"\broadmap ``({_ID})``"),
    re.compile(rf"\broadmap ({_ID})\b"),
    re.compile(rf"\bdocs/roadmap\.md ({_ID})\b"),
    re.compile(r"roadmap\.md#([xfrp]\d+[a-z]?)-"),
)

#: The provenance and family uses of ``P1`` a page may keep: the P1 plan and its PRs and
#: parts, the support families P1 and P2, X83's probe P1, a study's primary scenario and the
#: evidence pages' "fixed before the run in the P1 plan".
P1_ALLOWED = re.compile(
    r"P1 plan|P1 PR \d|P1 part|famil(?:y|ies) P1\b|P1 and P2|Probe P1|P1 primary"
)
#: The ``src/`` forms that claim P1 owns a gap. A bare ``"P1"`` is a support-row family.
P1_SOURCE_FORMS = re.compile(r"outside P1|roadmap ``P1``|roadmap=\"P1\"|reopened_by=\"P1\"")


def roadmap_headings() -> set[str]:
    """The ID of every heading in ``docs/roadmap.md``, and the stem of every red-cell owner.

    A red-cell owner such as ``X20-bootstrap`` is a row of the roadmap's owner table, not an
    item, and a text that names it by its stem ("roadmap X20") resolves to that row.
    """
    text = ROADMAP.read_text(encoding="utf-8")
    owners = re.findall(rf"^\| `({_ID})-[^`]*` \|", section(text, "Red-cell owners"), re.MULTILINE)
    return set(re.findall(rf"^#{{2,4}} ({_ID})\. ", text, re.MULTILINE)) | set(owners)


def unresolved_ids(text: str, headings: set[str]) -> list[str]:
    """Each roadmap ID that ``text`` names as an owner and that has no heading.

    A link to the roadmap at a pinned commit names the item as it stood then, so absolute
    URLs are dropped first.
    """
    text = re.sub(r"https?://\S+", "", text)
    found = []
    for form in OWNER_FORMS:
        for match in form.finditer(text):
            raw = match.group(1)
            item = raw[0].upper() + raw[1:]
            if item not in headings:
                found.append(f"{item}: {match.group(0)!r}")
    return found


def p1_page_findings(text: str) -> list[str]:
    """Each ``P1`` on a page that is not a provenance or family use."""
    flat = re.sub(r"\s+", " ", text)
    found = []
    for match in re.finditer(r"\bP1\b", flat):
        window = flat[max(0, match.start() - 12) : match.end() + 12]
        if not P1_ALLOWED.search(window):
            found.append(flat[max(0, match.start() - 60) : match.end() + 40])
    return found


def p1_source_findings(text: str) -> list[str]:
    """Each ``src/`` form that claims P1 owns a gap."""
    return [match.group(0) for match in P1_SOURCE_FORMS.finditer(text)]


def reader_text(path: Path) -> str:
    """A page's source text; a notebook's cell sources, without stored outputs."""
    if path.suffix == ".ipynb":
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        return "\n\n".join("".join(cell["source"]) for cell in cells)
    return path.read_text(encoding="utf-8")


def pages() -> list[Path]:
    """Every reader-facing page except the delivered list, which names removed items."""
    return [path for path in READER_FACING if path != DELIVERED]


def test_every_roadmap_item_a_source_names_as_an_owner_has_a_heading() -> None:
    headings = roadmap_headings()
    problems = [
        f"{path.relative_to(ROOT)}: {item}"
        for path in sorted(SRC.rglob("*.py"))
        for item in unresolved_ids(path.read_text(encoding="utf-8"), headings)
    ]
    assert not problems, "\n".join(problems)


def test_every_roadmap_item_a_page_names_as_an_owner_has_a_heading() -> None:
    headings = roadmap_headings()
    problems = [
        f"{path.relative_to(ROOT)}: {item}"
        for path in pages()
        for item in unresolved_ids(reader_text(path), headings)
    ]
    assert not problems, "\n".join(problems)


def test_no_page_names_p1_except_as_provenance() -> None:
    problems = [
        f"{path.relative_to(ROOT)}: ...{excerpt}..."
        for path in pages()
        for excerpt in p1_page_findings(reader_text(path))
    ]
    assert not problems, "\n".join(problems)


def test_no_source_claims_p1_owns_a_gap() -> None:
    problems = [
        f"{path.relative_to(ROOT)}: {form}"
        for path in sorted(SRC.rglob("*.py"))
        for form in p1_source_findings(path.read_text(encoding="utf-8"))
    ]
    assert not problems, "\n".join(problems)


def test_p1_has_left_the_roadmap_and_f42_has_one_row_and_one_contract() -> None:
    text = ROADMAP.read_text(encoding="utf-8")
    assert "P1" not in roadmap_headings()
    assert "#p1-" not in text
    assert len(re.findall(r"^### F42\. EP learner theory$", text, re.MULTILINE)) == 1
    rows = FUTURE_ROW.findall(section(text, "Future investigations"))
    assert rows.count("F42") == 1


#: The headings the self-tests resolve against, so they do not depend on the live roadmap.
SYNTHETIC_HEADINGS = {"X53", "F5", "X82"}


@pytest.mark.parametrize(
    "line",
    [
        "the gap is tracked (P1 in docs/roadmap.md).",
        "vector treatment (P1 in ``docs/roadmap.md``).",
        "P1 in the roadmap is the condition that reopens it",
        "An ``IMPLEMENTATION_GAP`` record with roadmap ``P1``.",
        "| refused | `implementation_gap`, roadmap P1 |",
        "docs/roadmap.md P1 tracks the point estimate.",
        "refuses under [P1](../roadmap.md#p1-ep-learner), because",
    ],
)
def test_the_generic_check_sees_each_form_of_a_removed_item(line: str) -> None:
    assert unresolved_ids(line, SYNTHETIC_HEADINGS)


@pytest.mark.parametrize(
    "line",
    [
        "X53 in docs/roadmap.md tracks this.",
        "open (roadmap X53).",
        "refuses under [F5](../roadmap.md#f5-other-refused-c-tmle-and-dr-tmle-compositions).",
        "An ``IMPLEMENTATION_GAP`` record with roadmap ``X82``.",
    ],
)
def test_the_generic_check_passes_an_item_with_a_heading(line: str) -> None:
    assert not unresolved_ids(line, SYNTHETIC_HEADINGS)


@pytest.mark.parametrize(
    "line",
    [
        "| refused | `implementation_gap`, roadmap P1 |",
        "a composition outside P1, a missing treatment",
        "every route refused under P1",
    ],
)
def test_the_page_check_sees_a_bare_p1(line: str) -> None:
    assert p1_page_findings(line)


@pytest.mark.parametrize(
    "line",
    [
        "R6 of the P1 plan",
        "(P1 PR 3)",
        "P1 part (a)",
        "The `T54` rows cover families P1 and P2.",
        "families P1 and P2 only",
        "Probe P1 (X83) measured the leak",
        "| P1 primary, two scenarios | 2,000 each |",
        "were fixed before the run in the P1\nplan's design.",
    ],
)
def test_the_page_check_passes_each_provenance_and_family_use(line: str) -> None:
    assert not p1_page_findings(line)


@pytest.mark.parametrize(
    "line",
    [
        "#: What each composition outside P1 is missing.",
        "An ``IMPLEMENTATION_GAP`` record with roadmap ``P1``.",
        '    roadmap="P1",',
        '    reopened_by="P1",',
    ],
)
def test_the_source_check_sees_each_ownership_form(line: str) -> None:
    assert p1_source_findings(line)


@pytest.mark.parametrize(
    "line",
    [
        "#: R-int-strata of the P1 plan: baseline strata beside an intermediate.",
        '    family="P1",',
        "the ATE (P1)",
    ],
)
def test_the_source_check_passes_provenance_and_the_support_family(line: str) -> None:
    assert not p1_source_findings(line)
