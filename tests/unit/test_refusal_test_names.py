"""Every test name that a refusal suggests must be a test that ``refute`` runs."""

from __future__ import annotations

import re
from pathlib import Path

import cleverly
from cleverly.validation.refute import _KNOWN_TESTS

_SUGGESTION = re.compile(r"tests\s*=\s*\(([^)]*)\)")
_NAME = re.compile(r"""['"]([A-Za-z_]+)['"]""")


def test_suggested_test_names_are_known_tests():
    root = Path(cleverly.__file__).parent
    found = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in _SUGGESTION.finditer(text):
            found.extend((path.name, name) for name in _NAME.findall(match.group(1)))
    assert found, "the scan found no tests=(...) suggestion, so the pattern is stale"
    unknown = [(file, name) for file, name in found if name not in _KNOWN_TESTS]
    assert not unknown


_COUNT = re.compile(
    r"\b(?:\d+|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
    r"(?:\w+\s+)?(?:operations|refuters|tests)\b",
    re.IGNORECASE,
)


def test_the_refuter_table_has_a_row_for_every_known_test():
    """X124 C2: the page counted seven operations over a nine-row table, against ten tests.

    Each name in ``_KNOWN_TESTS`` has a row in the refuter table of the validation-methods
    page, and the paragraph above the table states no count of operations.
    """
    page = Path(__file__).parents[2] / "docs/technical-reference/validation-methods.md"
    text = page.read_text(encoding="utf-8")
    header = "| refuter | what it does | statistic and reference | what it tests |"
    start = text.index(header)
    table = text[start : text.index("\n\n", start)]
    rows = {line.split("|")[1].strip().strip("`") for line in table.splitlines()[2:]}
    assert set(_KNOWN_TESTS) <= rows
    intro = text[text.rindex("**What each tells you.**", 0, start) : start]
    assert not _COUNT.findall(intro)
