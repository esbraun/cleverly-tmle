"""Where the documentation is, and how to read the Python out of it.

Two modules check the documentation and both need the same answer to "which files are the
documentation set": :mod:`tests.unit.test_documentation_links` and
:mod:`tests.unit.test_documentation_examples`.  Defining that twice is how the two drift --
a new guide added under ``docs/`` would be covered by one of them and not the other, and
nothing would say so.

Three modules now, since :mod:`tests.unit.test_documentation_runtime` *executes* the
reader-facing subset rather than only compiling it.  That does not make examples an
executable tier in the sense ``docs/architecture-invariants.md`` rules out: the runtime check
asserts that nothing raises and asserts nothing about any number, so an example is still
explanatory material and still not statistical evidence.

:data:`READER_FACING` is a *second* set and not a widening of the first.  The three modules
above ask "where is the Python", so they want markdown and nothing else;
:mod:`tests.unit.test_documentation_prose` asks "what does a reader read", which reaches a
notebook and an ``.rst`` file and stops short of the contributor instructions.  Collapsing the
two would make each answer wrong for the other.

This deliberately carries no section metadata, no dependency graph and no change selector.
The only thing wanted here is the text of each block and where it starts; which documents are
executable, and what each needs defined first, is that module's own registry.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = ["DOCUMENTS", "READER_FACING", "ROOT", "pipe_table", "python_blocks"]

ROOT = Path(__file__).resolve().parents[1]

#: Every markdown file the documentation set is made of.  ``docs/`` is the bulk; the
#: root-level files are linked from it and link back into it, so leaving them out would
#: check one direction of a two-way relationship.
DOCUMENTS = sorted(
    {
        *ROOT.glob("*.md"),
        *ROOT.glob("docs/**/*.md"),
    }
)

#: The prose scope ``CLAUDE.md`` declares: the root ``README.md`` and every reader-facing
#: source under ``docs/``.  Three ways this differs from :data:`DOCUMENTS`, each deliberate.
#:
#: It **excludes** ``CLAUDE.md`` and ``AGENTS.md``, which :data:`DOCUMENTS` picks up from the
#: root glob.  Those are instructions to a contributor rather than documentation, and
#: ``CLAUDE.md`` states the dash rule by quoting the characters it bans.
#:
#: It **adds** ``.rst`` and ``.ipynb``.  The notebook is the one that matters: it is in the
#: ``docs/examples`` toctree and reader-facing by every other measure, and an em dash survived a
#: whole sweep inside it because the checker of the day could not read a notebook at all.
#:
#: It **excludes** ``docs/api/generated/`` and ``docs/_build/``.  Both are build output, both
#: are gitignored, and ``CLAUDE.md`` says in as many words that they are not source.
READER_FACING = sorted(
    {
        ROOT / "README.md",
        *(
            path
            for suffix in ("md", "rst", "ipynb")
            for path in ROOT.glob(f"docs/**/*.{suffix}")
            if "_build" not in path.parts and "generated" not in path.parts
        ),
    }
)

#: A fenced ``python`` block.  Non-greedy to the closing fence, and anchored at line starts
#: so a fence quoted inside another block's body cannot open a match.
FENCE = re.compile(r"^```python\n(?P<code>.*?)^```", re.MULTILINE | re.DOTALL)


def python_blocks(document: Path) -> list[tuple[int, str]]:
    """Return ``(line number of the opening fence, code)`` for each block in ``document``."""
    text = document.read_text(encoding="utf-8")
    return [
        (text[: match.start()].count("\n") + 1, match.group("code"))
        for match in FENCE.finditer(text)
    ]


def pipe_table(
    document: Path, columns: tuple[str, ...], *, section: str | None = None
) -> list[dict[str, str]]:
    """The first pipe table in ``document`` whose header is exactly ``columns``.

    Selecting by header rather than by position is what lets a document carry several tables
    of different shapes without any of the gates over them counting lines.  A renamed column
    is then a failure here instead of a silent reinterpretation of every row beneath it.
    When ``section`` is supplied, search only the level-two heading with that text or anchor;
    method-evidence studies share one document and must not read one another's measured-value
    tables.
    """
    lines = document.read_text(encoding="utf-8").splitlines()
    if section is not None:

        def matches_section(line: str) -> bool:
            if not line.startswith("## "):
                return False
            heading = line[3:].strip()
            kept = "".join(
                character
                for character in heading.casefold()
                if character.isalnum() or character in " -_"
            )
            anchor = kept.strip().replace(" ", "-")
            return heading.casefold() == section.casefold() or anchor == section

        start = next(
            (index for index, line in enumerate(lines) if matches_section(line)),
            None,
        )
        assert start is not None, f"{document.name} has no level-two section {section!r}"
        stop = next(
            (
                index
                for index, line in enumerate(lines[start + 1 :], start=start + 1)
                if line.startswith("## ")
            ),
            len(lines),
        )
        lines = lines[start:stop]
    header = next(
        (
            index
            for index, line in enumerate(lines)
            if [cell.strip() for cell in line.strip().strip("|").split("|")] == list(columns)
        ),
        None,
    )
    assert header is not None, f"{document.name} has no table with the header {columns}"

    rows: list[dict[str, str]] = []
    for line in lines[header + 2 :]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        assert len(cells) == len(columns), f"ragged row in {document.name}: {line}"
        rows.append(dict(zip(columns, cells, strict=True)))
    return rows
