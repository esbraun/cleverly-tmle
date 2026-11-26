"""The examples index, the topic pages and the notebooks agree on where each tutorial lives.

The sidebar of the examples section shows one entry for each topic page, collapsed, and the
tutorials of a topic under it. The index repeats the topics as headings, in the same order, each
with the same one-paragraph scope and links to the same tutorials. A tutorial that joins no topic
would still build (Sphinx reports only a page in no toctree), but would vanish from the index's
topic lists, so these checks pin the three views together.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "docs" / "examples"
TOPICS = EXAMPLES / "topics"
INDEX = EXAMPLES / "index.md"

TOCTREE = re.compile(r"^```\{toctree\}\n(?P<body>.*?)^```", re.MULTILINE | re.DOTALL)
NOTEBOOK_LINK = re.compile(r"\]\((?:\.\./)?(?P<stem>[a-z0-9-]+)\.ipynb\)")


def toctree_entries(text: str) -> list[str]:
    """Entries of every toctree in ``text``, in order, options dropped."""
    entries: list[str] = []
    for match in TOCTREE.finditer(text):
        for line in match.group("body").splitlines():
            line = line.strip()
            if line and not line.startswith(":"):
                entries.append(line)
    return entries


def index_topic_order() -> list[str]:
    """Topic slugs in the order of the index toctree."""
    entries = toctree_entries(INDEX.read_text(encoding="utf-8"))
    assert all(entry.startswith("topics/") for entry in entries), entries
    return [entry.removeprefix("topics/") for entry in entries]


TITLED_ENTRY = re.compile(r"^(?P<label>.+) </examples/(?P<stem>[a-z0-9-]+)>$")


def topic_entries(slug: str) -> list[tuple[str, str]]:
    """(sidebar label, stem) of each entry of one topic page's toctree, in order."""
    entries = toctree_entries((TOPICS / f"{slug}.md").read_text(encoding="utf-8"))
    matches = [TITLED_ENTRY.match(entry) for entry in entries]
    assert all(matches), entries
    return [(match.group("label"), match.group("stem")) for match in matches if match]


def topic_stems(slug: str) -> list[str]:
    """Tutorial stems of one topic page's toctree, in order."""
    return [stem for _, stem in topic_entries(slug)]


def notebook_title(stem: str) -> str:
    """The level-one heading of an example notebook."""
    cells = json.loads((EXAMPLES / f"{stem}.ipynb").read_text(encoding="utf-8"))["cells"]
    for cell in cells:
        if cell["cell_type"] == "markdown":
            for line in "".join(cell["source"]).splitlines():
                if line.startswith("# "):
                    return line.removeprefix("# ")
    raise AssertionError(f"{stem} has no title")


def topic_title_and_scope(slug: str) -> tuple[str, str]:
    """The H1 and the first paragraph of a topic page, the paragraph unwrapped."""
    text = (TOPICS / f"{slug}.md").read_text(encoding="utf-8")
    title, _, rest = text.partition("\n\n")
    scope = rest.split("\n\n", 1)[0]
    return title.removeprefix("# "), " ".join(scope.split())


def index_sections() -> dict[str, tuple[str, str, list[str]]]:
    """Each topic heading of the index, keyed by slug: (title, scope, linked stems)."""
    text = INDEX.read_text(encoding="utf-8")
    body = text.split("## Find the example for your problem", 1)[1].split("\n## ", 1)[0]
    heading = re.compile(r"^### \[(?P<title>[^\]]+)\]\(topics/(?P<slug>[a-z0-9-]+)\.md\)$", re.M)
    matches = list(heading.finditer(body))
    sections: dict[str, tuple[str, str, list[str]]] = {}
    for match, after in zip(matches, [*matches[1:], None], strict=True):
        chunk = body[match.end() : after.start() if after else len(body)].strip()
        scope = " ".join(chunk.split("\n\n", 1)[0].split())
        stems = NOTEBOOK_LINK.findall(chunk)
        sections[match.group("slug")] = (match.group("title"), scope, stems)
    return sections


def test_every_example_notebook_is_in_exactly_one_topic() -> None:
    """Each notebook under docs/examples sits in one topic toctree, and every entry exists."""
    notebooks = sorted(path.stem for path in EXAMPLES.glob("*.ipynb"))
    placed = [stem for slug in index_topic_order() for stem in topic_stems(slug)]
    assert sorted(placed) == notebooks
    assert len(placed) == len(set(placed)), "a notebook sits in two topics"


def test_the_index_toctree_lists_every_topic_page() -> None:
    """No topic page is left out of the sidebar, and the toctree names no missing page."""
    assert sorted(index_topic_order()) == sorted(path.stem for path in TOPICS.glob("*.md"))


def test_the_index_headings_match_the_topic_pages() -> None:
    """Same topics, same order, same titles and scope, same tutorials as each topic page."""
    order = index_topic_order()
    sections = index_sections()
    assert list(sections) == order
    for slug in order:
        title, scope, stems = sections[slug]
        assert (title, scope) == topic_title_and_scope(slug), slug
        assert stems == topic_stems(slug), slug


def test_each_topic_table_has_one_row_per_tutorial_in_toctree_order() -> None:
    """The situation table of a topic page names its tutorials, in the toctree's order.

    A "See also" section may link tutorials of other topics, so only the table is compared.
    """
    for slug in index_topic_order():
        text = (TOPICS / f"{slug}.md").read_text(encoding="utf-8")
        table = text.split("| your situation | example | the mistake it shows |", 1)[1]
        table = table.split("\n\n", 1)[0]
        rows = [line for line in table.splitlines() if line.startswith("| [") or "](../" in line]
        linked = [NOTEBOOK_LINK.search(row.split(" | ")[1]).group("stem") for row in rows]
        assert linked == topic_stems(slug), slug


def test_the_methods_table_covers_each_tutorial_in_topic_order() -> None:
    """The methods table of the index follows the topics, one row per tutorial but plots."""
    text = INDEX.read_text(encoding="utf-8")
    table = text.split("## The methods behind each example", 1)[1].split("```{toctree}", 1)[0]
    stems = [m.group(1) for m in re.finditer(r"^\| \[[^\]]+\]\(([a-z0-9-]+)\.ipynb\)", table, re.M)]
    expected = [s for slug in index_topic_order() for s in topic_stems(slug) if s != "plots"]
    assert stems == expected


def test_each_sidebar_label_is_the_subject_half_of_the_title() -> None:
    """The sidebar shows the subject of each title and leaves the method to the page.

    A title reads "<subject>: <method>", or ends its subject with a question mark. Method names
    such as "longitudinal TMLE" mean nothing to a reader who is still choosing a tutorial, and
    they wrap most sidebar entries onto three lines.
    """
    for slug in index_topic_order():
        for label, stem in topic_entries(slug):
            title = notebook_title(stem)
            question = re.match(r"^(.*?\?) ", title)
            subject = question.group(1) if question else title.split(": ", 1)[0]
            assert label == subject, (stem, label, subject)
