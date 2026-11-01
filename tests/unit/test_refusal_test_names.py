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
