"""The user-guide index, its topic pages and its sidebar agree on where each page lives.

The sidebar of the user guide has captioned groups, one for each stage of an analysis and one for
the designs with their own pages. A group lists guide pages directly, or topic pages under
``docs/user-guide/topics/``, each a collapsed entry with its guide pages under it. The index
repeats the groups as level-three headings: a page listed directly is a bullet, and a topic is a
level-four heading with the topic page's one-paragraph scope and a bullet for each of its pages.
A guide page that joins no toctree would still build only with a warning, and one that joins two
would sit under two headings, so these checks pin the three views together and keep every page
under exactly one heading.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs" / "user-guide"
TOPICS = GUIDE / "topics"
INDEX = GUIDE / "index.md"

TOCTREE = re.compile(r"^```\{toctree\}\n(?P<body>.*?)^```", re.MULTILINE | re.DOTALL)
TITLED_ENTRY = re.compile(r"^(?P<label>.+) <(?:/user-guide/)?(?P<target>[a-z0-9/-]+)>$")
GROUP_HEADING = re.compile(r"^### (?P<caption>.+)$", re.M)
TOPIC_HEADING = re.compile(r"^#### \[(?P<title>[^\]]+)\]\(topics/(?P<slug>[a-z0-9-]+)\.md\)$", re.M)
BULLET = re.compile(r"^- \[(?P<label>[^\]]+)\]\((?P<stem>[a-z0-9-]+)\.md\)", re.M)


def toctrees(text: str) -> list[tuple[str | None, list[tuple[str, str]]]]:
    """Each toctree in ``text`` as (caption or None, [(label, target)]), options dropped."""
    trees: list[tuple[str | None, list[tuple[str, str]]]] = []
    for match in TOCTREE.finditer(text):
        caption: str | None = None
        entries: list[tuple[str, str]] = []
        for line in match.group("body").splitlines():
            line = line.strip()
            if line.startswith(":caption:"):
                caption = line.removeprefix(":caption:").strip()
            elif line and not line.startswith(":"):
                found = TITLED_ENTRY.match(line)
                assert found, f"an untitled or malformed toctree entry: {line!r}"
                entries.append((found.group("label"), found.group("target")))
        trees.append((caption, entries))
    return trees


def index_groups() -> list[tuple[str, list[tuple[str, str]]]]:
    """(caption, entries) of each sidebar group, in order."""
    groups = []
    for caption, entries in toctrees(INDEX.read_text(encoding="utf-8")):
        assert caption, f"an index toctree without a caption: {entries}"
        groups.append((caption, entries))
    return groups


def topic_entries(slug: str) -> list[tuple[str, str]]:
    """(label, stem) of each guide page in one topic page's toctree, in order."""
    trees = toctrees((TOPICS / f"{slug}.md").read_text(encoding="utf-8"))
    entries = [entry for _, tree in trees for entry in tree]
    for _, target in entries:
        assert "/" not in target, f"topic {slug} lists {target}, which is not a guide page"
    return entries


def sidebar_pages() -> list[str]:
    """Every guide page stem in sidebar order, topic pages replaced by their pages."""
    pages: list[str] = []
    for _, entries in index_groups():
        for _, target in entries:
            if target.startswith("topics/"):
                pages.extend(stem for _, stem in topic_entries(target.removeprefix("topics/")))
            else:
                pages.append(target)
    return pages


def unwrap(paragraph: str) -> str:
    """A Markdown paragraph on one line, link targets dropped.

    The index and a topic page sit in different folders, so one link is spelled two ways.
    """
    return re.sub(r"\]\([^)\s]*\)", "]", " ".join(paragraph.split()))


def topic_title_and_scope(slug: str) -> tuple[str, str]:
    """The H1 and the first paragraph of a topic page, the paragraph unwrapped."""
    text = (TOPICS / f"{slug}.md").read_text(encoding="utf-8")
    title, _, rest = text.partition("\n\n")
    return title.removeprefix("# "), unwrap(rest.split("\n\n", 1)[0])


def index_finder() -> str:
    """The body of the index section "Find the page for your task"."""
    text = INDEX.read_text(encoding="utf-8")
    return text.split("## Find the page for your task", 1)[1].split("```{toctree}", 1)[0]


def split_on(heading: re.Pattern[str], text: str) -> list[tuple[re.Match[str], str]]:
    """Each match of ``heading`` in ``text`` with the text up to the next match."""
    matches = list(heading.finditer(text))
    ends = [match.start() for match in matches[1:]] + [len(text)] * bool(matches)
    return [(match, text[match.end() : end]) for match, end in zip(matches, ends, strict=True)]


def test_every_guide_page_sits_under_exactly_one_heading() -> None:
    """Each guide page appears once in the sidebar, and every entry names a page that exists."""
    placed = sidebar_pages()
    pages = sorted(path.stem for path in GUIDE.glob("*.md") if path.stem != "index")
    assert sorted(placed) == pages
    assert len(placed) == len(set(placed)), "a guide page sits under two headings"


def test_every_topic_page_is_in_the_sidebar_once() -> None:
    """No topic page is left out of the index toctrees, or listed twice."""
    listed = [
        target.removeprefix("topics/")
        for _, entries in index_groups()
        for _, target in entries
        if target.startswith("topics/")
    ]
    assert sorted(listed) == sorted(path.stem for path in TOPICS.glob("*.md"))
    assert len(listed) == len(set(listed))


def test_the_index_headings_match_the_sidebar() -> None:
    """Same groups, pages, topics, labels and scopes, in the same order, as the sidebar."""
    sections = split_on(GROUP_HEADING, index_finder())
    assert [match.group("caption") for match, _ in sections] == [c for c, _ in index_groups()]
    for (_, body), (caption, entries) in zip(sections, index_groups(), strict=True):
        topics = split_on(TOPIC_HEADING, body)
        direct = body if not topics else body[: body.index(topics[0][0].group(0))]
        seen: list[tuple[str, str]] = [
            (bullet.group("label"), bullet.group("stem")) for bullet in BULLET.finditer(direct)
        ]
        for match, chunk in topics:
            slug = match.group("slug")
            scope = unwrap(chunk.strip().split("\n\n", 1)[0])
            assert (match.group("title"), scope) == topic_title_and_scope(slug), slug
            bullets = [(b.group("label"), b.group("stem")) for b in BULLET.finditer(chunk)]
            assert bullets == topic_entries(slug), slug
            seen.append((match.group("title"), f"topics/{slug}"))
        assert seen == entries, caption


def test_each_topic_table_has_one_row_per_page_in_toctree_order() -> None:
    """The question table of a topic page links its pages, in the toctree's order."""
    row = re.compile(r"^\| .+ \| \[[^\]]+\]\(\.\./(?P<stem>[a-z0-9-]+)\.md\) \|$", re.M)
    for path in sorted(TOPICS.glob("*.md")):
        text = path.read_text(encoding="utf-8").split("```{toctree}", 1)[0]
        linked = [match.group("stem") for match in row.finditer(text)]
        assert linked == [stem for _, stem in topic_entries(path.stem)], path.stem
