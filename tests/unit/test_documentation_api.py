"""The production site exposes its promised sections and the supported root API."""

from __future__ import annotations

import re

import cleverly
import cleverly.datasets
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


def test_every_dataset_generator_is_in_the_python_api() -> None:
    """A shipped generator a reader cannot find is a generator that does not exist to them.

    Narrowed to the ``make_*`` prefix deliberately.  ``cleverly.datasets.__all__`` also exports the
    DGP objects and truth helpers the statistical studies are built from, which are machinery
    rather than a documented surface; the generators are the half a reader is meant to call.
    ``make_heterogeneous`` and ``make_instrument`` were both missing when this was added, because
    :func:`test_every_root_export_is_in_the_python_api` only reaches ``cleverly.__all__`` and a
    submodule can drift underneath it.
    """
    source = "\n".join(path.read_text(encoding="utf-8") for path in sorted(API_ROOT.glob("*.md")))
    generators = [name for name in cleverly.datasets.__all__ if name.startswith("make_")]
    assert generators, "no make_* generators found; check cleverly.datasets.__all__"
    missing = [name for name in generators if f"cleverly.datasets.{name}" not in source]
    assert not missing, f"dataset generators missing from docs/api: {missing}"


def test_the_categorised_pages_and_the_object_index_agree() -> None:
    """The two lists that have to be the same list, kept the same by a test.

    **This is load-bearing and not tidiness.**  ``sphinx.ext.autosummary`` generates its stub pages
    by scanning sources for the reStructuredText ``.. autosummary::`` form, and it does not read
    MyST's ``{autosummary}`` fence: pointed at the nine categorised ``.md`` pages it finds *zero*
    entries, and at ``object-index.rst`` it finds all of them.  So the ``.rst`` is what actually
    generates every object page, and the ``.md`` pages only render tables that link to them.

    An object added to a ``.md`` page alone therefore links to a stub that was never written, and
    the warning-as-error build fails somewhere unrelated-looking.  One added to the ``.rst`` alone
    generates a page no categorised table lists.  Naming both halves here turns either mistake into
    a message that says which file is short.
    """
    pattern = re.compile(r"^\s*(cleverly\.[A-Za-z0-9_.]+)\s*$", re.MULTILINE)
    categorised = {
        name
        for path in sorted(API_ROOT.glob("*.md"))
        for name in pattern.findall(path.read_text(encoding="utf-8"))
    }
    indexed = set(pattern.findall((API_ROOT / "object-index.rst").read_text(encoding="utf-8")))

    assert categorised, "no objects found on the categorised API pages; check the pattern"
    assert categorised == indexed, (
        f"docs/api/*.md and docs/api/object-index.rst disagree. "
        f"Only on a categorised page (no stub will be generated): {sorted(categorised - indexed)}. "
        f"Only in the object index (no table lists it): {sorted(indexed - categorised)}"
    )
