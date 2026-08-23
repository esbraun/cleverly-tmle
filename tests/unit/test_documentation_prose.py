r"""No reader-facing source joins two clauses with an em dash or a double hyphen.

``CLAUDE.md`` states the rule and ``.vale.ini`` is meant to enforce it.  This module exists
because Vale enforced nothing, and because a green Vale run and a broken one are the same
output.

**The failure this module was written after.**  ``Cleverly.ClauseDash`` extends Vale's
``existence`` rule, which wraps every token in a word boundary.  An em dash is not a word
character, so ``\b—\b`` can never be satisfied: the rule matched ``' -- '`` -- whose spaces sit
next to word characters -- and silently matched no em dash anywhere.  The sweep that added the
rule reported ``0 errors`` over the whole tree on a config that could not have failed.  That is
the shape ``CLAUDE.md`` warns about: a check that cannot fire reads exactly like a check that
passed.

**Two things Vale still cannot do, which is the other half of why this is a test.**  It has no
notebook reader, and ``docs/examples/twins-causal-inference.ipynb`` is in the ``docs/examples``
toctree -- an em dash survived the whole sweep in one of its headings.  And it needs docutils'
``rst2html`` for a single ``.rst`` file, which is a whole optional extra in CI for two lines of
prose.  :data:`tests.documents.READER_FACING` reaches all three formats with no external
binary, so the two checks overlap on markdown and neither is the only cover for anything.

**What is deliberately not checked.**  Code is exempt, in all three of its forms: a fenced
block, an inline span, and a notebook's code cells and their outputs.  Each is a real source of
the characters -- ``git diff origin/main -- docs`` is a command, ``--`` is how ``CLAUDE.md``
names the thing it bans, and a rendered dataframe draws its rules with runs of hyphens.  This
mirrors Vale's ``scope: text``.  Sentence and paragraph length stay with Vale, which measures
them properly; this module checks the one rule that is exact.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.documents import READER_FACING, ROOT

#: The two forms ``CLAUDE.md`` bans, and the same two tokens ``Cleverly.ClauseDash`` carries.
#: The double hyphen is spaced because a bare ``--`` is also a command-line flag, a YAML
#: document marker and a markdown table rule.
DASHES = ("—", " -- ")

#: A fence open or close, matched the way :mod:`tests.unit.test_documentation_links` matches
#: one so the two modules agree on what "inside a block" means.
FENCE = re.compile(r"^\s*(```|~~~)")

#: An inline code span, non-greedy so ``a `b` and `c` d`` is two spans rather than one.
#: Doubled backticks first, since ```` ``a`b`` ```` is one span and not three.
CODE_SPAN = re.compile(r"``[^`]+``|`[^`]+`")


def prose_lines(text: str) -> list[tuple[int, str]]:
    """``(line number, text)`` for each line of prose, with code removed.

    Fenced blocks are dropped whole and inline spans are blanked in place rather than deleted,
    so a reported column still means something on the line the reader sees.
    """
    out: list[tuple[int, str]] = []
    fenced = False
    for number, line in enumerate(text.splitlines(), start=1):
        if FENCE.match(line):
            fenced = not fenced
            continue
        if fenced:
            continue
        out.append((number, CODE_SPAN.sub(lambda match: " " * len(match.group()), line)))
    return out


def markdown_cells(notebook: Path) -> str:
    """Every markdown cell of ``notebook``, joined.

    Code cells and their outputs are skipped, and that is not tidiness.  A rendered table in a
    stored output draws its rules with hyphens -- ``----------  ------  --  ---`` appears in the
    TWINS notebook -- and every run of them contains the spaced double hyphen this module bans.
    """
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    return "\n".join(
        "".join(cell["source"]) for cell in cells if cell.get("cell_type") == "markdown"
    )


def offences(document: Path) -> list[str]:
    """Every banned dash in ``document``'s prose, as ``path:line: text``."""
    if document.suffix == ".ipynb":
        text = markdown_cells(document)
    else:
        text = document.read_text(encoding="utf-8")
    return [
        f"{document.relative_to(ROOT)}:{number}: {line.strip()}"
        for number, line in prose_lines(text)
        if any(dash in line for dash in DASHES)
    ]


#: A floor on the set, so a glob that stopped matching cannot report success over nothing.
#: Well under the count at the time of writing, because this is a guard and not a census.
_EXPECTED_DOCUMENTS = 40


def test_the_sweep_reaches_something() -> None:
    """The check below is only as good as the set it runs over."""
    assert len(READER_FACING) >= _EXPECTED_DOCUMENTS, (
        f"only {len(READER_FACING)} reader-facing sources found; check documents.READER_FACING"
    )
    suffixes = {path.suffix for path in READER_FACING}
    assert suffixes == {".md", ".rst", ".ipynb"}, (
        f"the sweep reaches {sorted(suffixes)}; Vale covers markdown alone, so a format "
        "dropped here is a format nothing checks"
    )


def test_the_scanner_sees_a_planted_dash() -> None:
    """A deliberate mutation, because every assertion below passes on an empty result.

    This is the control the Vale rule did not have.  Each case fails a different way: an em
    dash that stopped matching, a spaced double hyphen that stopped matching, a fence or span
    exclusion that grew to swallow prose, and a notebook reader that returned nothing.
    """
    caught = prose_lines("An em dash — here.\nA flag -- here.\n")
    assert [number for number, line in caught if any(d in line for d in DASHES)] == [1, 2]

    exempt = prose_lines("```\ngit diff main -- docs\n```\nAn inline `--` span.\n")
    assert not [line for _, line in exempt if any(dash in line for dash in DASHES)]


def test_the_notebook_reader_skips_output() -> None:
    """Markdown cells in, code cells and stored outputs out."""
    notebooks = [path for path in READER_FACING if path.suffix == ".ipynb"]
    assert notebooks, "no notebook in the reader-facing set; check documents.READER_FACING"
    for notebook in notebooks:
        cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
        assert any(cell.get("cell_type") == "markdown" for cell in cells), notebook
        assert any(cell.get("cell_type") == "code" for cell in cells), notebook
        assert "import" not in markdown_cells(notebook).split("```")[0]


@pytest.mark.parametrize(
    "document", READER_FACING, ids=lambda path: str(path.relative_to(ROOT).as_posix())
)
def test_no_clause_is_joined_by_a_dash(document: Path) -> None:
    """One idea per sentence, and a full stop or a table where a dash was standing."""
    found = offences(document)
    assert not found, "\n".join(
        ["join these clauses with a full stop, a comma, parentheses or a table:", *found]
    )
