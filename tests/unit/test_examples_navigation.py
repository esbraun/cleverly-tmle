"""The examples index, the topic pages and the notebooks agree on where each tutorial lives.

The sidebar of the examples section has a few captioned groups. The first, "Start here", lists
two tutorials directly. Each other group lists topic pages, collapsed, with the tutorials of a
topic under it. The index repeats the groups as level-three headings and the topics as
level-four headings, in the same order, each topic with the same one-paragraph scope and links to
the same tutorials. A tutorial that joins no topic would still build (Sphinx reports only a page
in no toctree), but would vanish from the index's topic lists, so these checks pin the three views
together.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "docs" / "examples"
TOPICS = EXAMPLES / "topics"
INDEX = EXAMPLES / "index.md"

#: The caption of the one index toctree that lists tutorials rather than topic pages.
START_CAPTION = "Start here"

TOCTREE = re.compile(r"^```\{toctree\}\n(?P<body>.*?)^```", re.MULTILINE | re.DOTALL)
NOTEBOOK_LINK = re.compile(r"\]\((?:\.\./)?(?P<stem>[a-z0-9-]+)\.ipynb\)")
TITLED_ENTRY = re.compile(r"^(?P<label>.+) <(?:/examples/)?(?P<stem>[a-z0-9-]+)>$")


def toctrees(text: str) -> list[tuple[str | None, list[str]]]:
    """Each toctree in ``text`` as (caption or None, entries), options dropped."""
    trees: list[tuple[str | None, list[str]]] = []
    for match in TOCTREE.finditer(text):
        caption: str | None = None
        entries: list[str] = []
        for line in match.group("body").splitlines():
            line = line.strip()
            if line.startswith(":caption:"):
                caption = line.removeprefix(":caption:").strip()
            elif line and not line.startswith(":"):
                entries.append(line)
        trees.append((caption, entries))
    return trees


def titled(entries: list[str]) -> list[tuple[str, str]]:
    """(sidebar label, stem) of each titled toctree entry."""
    matches = [TITLED_ENTRY.match(entry) for entry in entries]
    assert all(matches), entries
    return [(match.group("label"), match.group("stem")) for match in matches if match]


def index_groups() -> list[tuple[str, list[str]]]:
    """(caption, topic slugs) of each topic group of the index, in order."""
    trees = toctrees(INDEX.read_text(encoding="utf-8"))
    assert trees[0][0] == START_CAPTION, trees[0]
    groups: list[tuple[str, list[str]]] = []
    for caption, entries in trees[1:]:
        assert caption, entries
        assert all(entry.startswith("topics/") for entry in entries), entries
        groups.append((caption, [entry.removeprefix("topics/") for entry in entries]))
    return groups


def start_entries() -> list[tuple[str, str]]:
    """(sidebar label, stem) of the tutorials listed directly under "Start here"."""
    return titled(toctrees(INDEX.read_text(encoding="utf-8"))[0][1])


def index_topic_order() -> list[str]:
    """Topic slugs in sidebar order."""
    return [slug for _, slugs in index_groups() for slug in slugs]


def topic_entries(slug: str) -> list[tuple[str, str]]:
    """(sidebar label, stem) of each entry of one topic page's toctree, in order."""
    trees = toctrees((TOPICS / f"{slug}.md").read_text(encoding="utf-8"))
    entries = [entry for _, tree in trees for entry in tree]
    assert all(entry.split(" <", 1)[-1].startswith("/examples/") for entry in entries), entries
    return titled(entries)


def topic_stems(slug: str) -> list[str]:
    """Tutorial stems of one topic page's toctree, in order."""
    return [stem for _, stem in topic_entries(slug)]


def sidebar_order() -> list[str]:
    """Every tutorial stem in the order the sidebar lists it."""
    return [stem for _, stem in start_entries()] + [
        stem for slug in index_topic_order() for stem in topic_stems(slug)
    ]


def notebook_title(stem: str) -> str:
    """The level-one heading of an example notebook."""
    cells = json.loads((EXAMPLES / f"{stem}.ipynb").read_text(encoding="utf-8"))["cells"]
    for cell in cells:
        if cell["cell_type"] == "markdown":
            for line in "".join(cell["source"]).splitlines():
                if line.startswith("# "):
                    return line.removeprefix("# ")
    raise AssertionError(f"{stem} has no title")


def unwrap(paragraph: str) -> str:
    """A Markdown paragraph on one line."""
    return " ".join(paragraph.split())


def topic_title_and_scope(slug: str) -> tuple[str, str]:
    """The H1 and the first paragraph of a topic page, the paragraph unwrapped."""
    text = (TOPICS / f"{slug}.md").read_text(encoding="utf-8")
    title, _, rest = text.partition("\n\n")
    return title.removeprefix("# "), unwrap(rest.split("\n\n", 1)[0])


def index_finder() -> str:
    """The body of the index section "Find the example for your problem"."""
    text = INDEX.read_text(encoding="utf-8")
    return text.split("## Find the example for your problem", 1)[1].split("\n## ", 1)[0]


def split_on(heading: re.Pattern[str], text: str) -> list[tuple[re.Match[str], str]]:
    """Each match of ``heading`` in ``text`` with the text up to the next match."""
    matches = list(heading.finditer(text))
    ends = [match.start() for match in matches[1:]] + [len(text)]
    return [(match, text[match.end() : end]) for match, end in zip(matches, ends, strict=True)]


GROUP_HEADING = re.compile(r"^### (?P<caption>.+)$", re.M)
TOPIC_HEADING = re.compile(r"^#### \[(?P<title>[^\]]+)\]\(topics/(?P<slug>[a-z0-9-]+)\.md\)$", re.M)


def test_every_example_notebook_is_listed_exactly_once() -> None:
    """Each notebook under docs/examples sits once in the sidebar, and every entry exists."""
    placed = sidebar_order()
    assert sorted(placed) == sorted(path.stem for path in EXAMPLES.glob("*.ipynb"))
    assert len(placed) == len(set(placed)), "a notebook sits in two places"


def test_the_index_toctrees_list_every_topic_page() -> None:
    """No topic page is left out of the sidebar, and no toctree names a missing page."""
    assert sorted(index_topic_order()) == sorted(path.stem for path in TOPICS.glob("*.md"))


def test_the_index_headings_match_the_sidebar() -> None:
    """Same groups, topics, titles, scopes and tutorials, in the same order, as the sidebar."""
    sections = split_on(GROUP_HEADING, index_finder())
    captions = [match.group("caption") for match, _ in sections]
    assert captions == [START_CAPTION] + [caption for caption, _ in index_groups()]
    start_body = sections[0][1]
    assert NOTEBOOK_LINK.findall(start_body) == [stem for _, stem in start_entries()]
    for (_, body), (caption, slugs) in zip(sections[1:], index_groups(), strict=True):
        topics = split_on(TOPIC_HEADING, body)
        assert [match.group("slug") for match, _ in topics] == slugs, caption
        for match, chunk in topics:
            slug = match.group("slug")
            scope = unwrap(chunk.strip().split("\n\n", 1)[0])
            assert (match.group("title"), scope) == topic_title_and_scope(slug), slug
            assert NOTEBOOK_LINK.findall(chunk) == topic_stems(slug), slug


def test_each_topic_table_has_one_row_per_tutorial_in_toctree_order() -> None:
    """The situation table of a topic page names its tutorials, in the toctree's order.

    A "See also" section may link tutorials of other topics, so only the table is compared.
    """
    for slug in index_topic_order():
        text = (TOPICS / f"{slug}.md").read_text(encoding="utf-8")
        table = text.split("| your situation | example | the mistake it shows |", 1)[1]
        rows = [line for line in table.split("\n\n", 1)[0].splitlines() if "](../" in line]
        linked = []
        for row in rows:
            match = NOTEBOOK_LINK.search(row.split(" | ")[1])
            assert match, f"{slug}: a table row links no notebook in its example column: {row}"
            linked.append(match.group("stem"))
        assert linked == topic_stems(slug), slug


def test_the_methods_table_covers_each_tutorial_in_sidebar_order() -> None:
    """The methods table of the index follows the sidebar, one row per tutorial but plots."""
    text = INDEX.read_text(encoding="utf-8")
    table = text.split("## The methods behind each example", 1)[1].split("```{toctree}", 1)[0]
    row = re.compile(r"^\| \[[^\]]+\]\(([a-z0-9-]+)\.ipynb\)", re.M)
    stems = [match.group(1) for match in row.finditer(table)]
    assert stems == [stem for stem in sidebar_order() if stem != "plots"]


def test_each_sidebar_label_is_the_subject_half_of_the_title() -> None:
    """The sidebar shows the subject of each title and leaves the method to the page.

    A title reads "<subject>: <method>", or ends its subject with a question mark. Method names
    such as "longitudinal TMLE" mean nothing to a reader who is still choosing a tutorial, and
    they wrap most sidebar entries onto three lines.
    """
    labelled = start_entries() + [
        pair for slug in index_topic_order() for pair in topic_entries(slug)
    ]
    for label, stem in labelled:
        title = notebook_title(stem)
        question = re.match(r"^(.*?\?) ", title)
        subject = question.group(1) if question else title.split(": ", 1)[0]
        assert label == subject, (stem, label, subject)


def test_every_full_title_link_matches_its_notebook() -> None:
    """Index bullets and See also lines quote each notebook's title exactly."""
    full = re.compile(r"^- \[(?P<text>[^\]]+)\]\((?:\.\./)?(?P<stem>[a-z0-9-]+)\.ipynb\)", re.M)
    for page in [INDEX, *TOPICS.glob("*.md")]:
        for match in full.finditer(page.read_text(encoding="utf-8")):
            assert match.group("text") == notebook_title(match.group("stem")), (page.name, match[0])
