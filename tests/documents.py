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

This deliberately carries no section metadata, no dependency graph and no change selector.
The only thing wanted here is the text of each block and where it starts; which documents are
executable, and what each needs defined first, is that module's own registry.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = ["DOCUMENTS", "ROOT", "python_blocks"]

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
