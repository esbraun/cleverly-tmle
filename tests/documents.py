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

__all__ = [
    "DOCUMENTS",
    "EXCLUDED_DIRECTORIES",
    "NOTEBOOKS",
    "READER_FACING",
    "ROOT",
    "pipe_table",
    "python_blocks",
    "reader_facing",
]

ROOT = Path(__file__).resolve().parents[1]

#: Directory names that hold generated or autosaved copies rather than source.  Matched against
#: every path component, so the depth of the directory does not matter.
EXCLUDED_DIRECTORIES = frozenset({"_build", "generated", ".ipynb_checkpoints"})

#: Every markdown file the documentation set is made of.  ``docs/`` is the bulk; the
#: root-level files are linked from it and link back into it, so leaving them out would
#: check one direction of a two-way relationship.  :data:`EXCLUDED_DIRECTORIES` applies here
#: for the same reason it applies below: a build directory and an autosave directory are
#: copies, and checking a copy checks nothing.
DOCUMENTS = sorted(
    {
        *ROOT.glob("*.md"),
        *(
            path
            for path in ROOT.glob("docs/**/*.md")
            if not EXCLUDED_DIRECTORIES.intersection(path.parts)
        ),
    }
)


def reader_facing(root: Path) -> list[Path]:
    """Every reader-facing source below ``root``, sorted, with the generated copies dropped.

    Three ways this differs from :data:`DOCUMENTS`, each deliberate.

    It **excludes** ``CLAUDE.md`` and ``AGENTS.md``, which :data:`DOCUMENTS` picks up from the
    root glob.  Those are instructions to a contributor rather than documentation, and
    ``CLAUDE.md`` states the dash rule by quoting the characters it bans.

    It **adds** ``.rst`` and ``.ipynb``.  The notebook is the one that matters: it is in the
    ``docs/examples`` toctree and reader-facing by every other measure, and an em dash survived
    a whole sweep inside it because the checker of the day could not read a notebook at all.

    It **excludes** ``docs/api/generated/``, ``docs/_build/`` and ``.ipynb_checkpoints/``.  The
    first two are build output, and ``CLAUDE.md`` says in as many words that they are not
    source.  The third is Jupyter's autosave copy.  ``Path.glob`` walks a dot-directory like
    any other, so opening the published notebook once would otherwise put an untracked copy of
    it in this set and fail the fast tier on a file nobody wrote.

    The root is a parameter so a test can run this selection over a scratch tree and check what
    it drops.  :data:`READER_FACING` is the one call the repository itself makes.
    """
    return sorted(
        {
            root / "README.md",
            *(
                path
                for suffix in ("md", "rst", "ipynb")
                for path in root.glob(f"docs/**/*.{suffix}")
                if not EXCLUDED_DIRECTORIES.intersection(path.parts)
            ),
        }
    )


#: The prose scope ``CLAUDE.md`` declares: the root ``README.md`` and every reader-facing
#: source under ``docs/``.  :func:`reader_facing` says what it keeps and what it drops.
READER_FACING = reader_facing(ROOT)

#: Every committed notebook that Sphinx can publish. Keeping this selection beside
#: :data:`READER_FACING` makes prose, runtime, and artifact checks discover future notebooks
#: from one definition instead of naming the current notebook in each consumer.
NOTEBOOKS = tuple(path for path in READER_FACING if path.suffix == ".ipynb")

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
    When ``section`` is supplied, search only the level-one heading with that text or anchor.
    Each method-evidence study is now its own page, so the range runs to end of file; the
    scoping stays because it is what makes the caller name the page it means to read.
    """
    lines = document.read_text(encoding="utf-8").splitlines()
    if section is not None:

        def matches_section(line: str) -> bool:
            if not line.startswith("# "):
                return False
            heading = line[2:].strip()
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
        assert start is not None, f"{document.name} has no level-one section {section!r}"
        stop = next(
            (
                index
                for index, line in enumerate(lines[start + 1 :], start=start + 1)
                if line.startswith("# ")
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
