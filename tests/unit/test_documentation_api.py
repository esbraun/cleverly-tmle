"""The production site exposes its promised sections and the supported root API."""

from __future__ import annotations

import cleverly
from tests.documents import ROOT

SITE_INDEX = ROOT / "docs" / "index.md"
API_ROOT = ROOT / "docs" / "api"


def test_site_navigation_has_every_public_section() -> None:
    """The six reader-facing sections are first-class navigation entries."""
    text = SITE_INDEX.read_text(encoding="utf-8")
    required = {
        "getting-started/index",
        "workflow",
        "user-guide/index",
        "technical-reference/index",
        "examples/index",
        "api/index",
    }
    assert required <= set(text.split())


def test_every_root_export_is_in_the_python_api() -> None:
    """A root public symbol cannot silently disappear from generated API source."""
    source = "\n".join(path.read_text(encoding="utf-8") for path in sorted(API_ROOT.glob("*.md")))
    missing = [name for name in cleverly.__all__ if f"cleverly.{name}" not in source]
    assert not missing, f"root exports missing from docs/api: {missing}"
