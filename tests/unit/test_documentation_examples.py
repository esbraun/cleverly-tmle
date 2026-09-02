r"""Every fenced ``python`` example in the documentation parses.

**Not an executable tier, and deliberately not.**  Documentation examples are explanatory
material rather than correctness evidence: behavior belongs in the fast suite or in a registered
validation study (``docs/architecture-invariants.md``). Compiling a block is
not running it -- no import is executed, no estimator is fitted, no fixture is needed -- so
this stays a static check of the prose while restoring the one guarantee that was lost with
the executable tier.

**Why ``ruff`` does not already cover this, contrary to what it looks like.**  ``ruff
format`` does reach inside fenced ``python`` blocks in markdown, which is why the formatter
is pinned exactly.  But the formatter *skips a block it cannot parse* rather than failing
on it: a fence containing ``this is not python(((`` passes ``ruff format --check``, while a
merely mis-formatted valid block fails it.  And the ``ruff`` **linter** does not read
markdown at all -- pointed at ``README.md`` it reports "No Python files found" and exits
zero.  So a syntax error in an example is invisible to both halves of the toolchain, and
renders to the reader as an ordinary example.

That is not hypothetical.  This module was added after a prose line in a planning document was retagged
from ``text`` to ``python`` and shipped through a green ``ruff check .``,
``ruff format --check .`` and full test run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.documents import DOCUMENTS, ROOT, python_blocks

#: The documents carrying any Python at all, computed rather than listed so that a new guide
#: is covered by existing rather than by being remembered.
WITH_CODE = [document for document in DOCUMENTS if python_blocks(document)]

#: A floor on what the sweep is expected to find.  Well under the count at the time of
#: writing, because this is not a census -- it is the one way a check over a discovered set
#: fails open.  A :data:`tests.documents.FENCE` that stopped matching would otherwise report
#: success over an empty set, which reads exactly like every example being fine.
_EXPECTED_BLOCKS = 30


def test_the_extractor_found_something() -> None:
    """The check below is only as good as the set it runs over."""
    assert WITH_CODE, "no markdown file in the tree has a ```python block; check documents.FENCE"
    found = sum(len(python_blocks(document)) for document in WITH_CODE)
    assert found >= _EXPECTED_BLOCKS, f"found only {found} python blocks; check documents.FENCE"


@pytest.mark.parametrize("document", WITH_CODE, ids=lambda path: str(path.relative_to(ROOT)))
def test_every_python_block_parses(document: Path) -> None:
    """A block tagged ``python`` is a promise that it is Python, kept by the suite."""
    for line, code in python_blocks(document):
        name = f"{document.relative_to(ROOT)}:{line}"
        try:
            compile(code, name, "exec")
        except SyntaxError as error:  # pragma: no cover - the failure is the message
            pytest.fail(f"{name} does not parse: {error}. Tag prose fences ```text, not ```python")
