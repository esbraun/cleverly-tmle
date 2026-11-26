"""The docs map covers the repository and names only pages that exist.

:mod:`tests.docs_map` routes a change to the pages it can make stale, and the ``docs-impact``
pull request check fails when a routed page is neither updated nor waived. A map with a hole
lets a change through unreviewed, and a map with a dead entry routes nothing, so this module
holds it to the tree both ways:

* every file in :data:`~tests.docs_map.IN_SCOPE` matches an area or ``INTERNAL``;
* every glob matches a tracked file, and every page exists as a documentation source;
* every reader-facing page is reachable from some area, or is listed as standalone with a reason;
* no module that defines a published name hides in ``INTERNAL``, and ``PUBLISHED`` lists every
  module the API pages publish;
* each ``guards`` entry is a test file that exists.

Each rule runs through a function that takes the file list and the map, and a doctored input
shows each one failing, so none of them can pass by checking nothing.
"""

from __future__ import annotations

import ast
import fnmatch
import re
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import pytest

from tests.docs_impact import exports
from tests.docs_map import AREAS, IN_SCOPE, INTERNAL, PUBLISHED, STANDALONE_PAGES, Area
from tests.documents import READER_FACING, ROOT

TRACKED = tuple(
    subprocess.run(
        ["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
)
#: Study pages are generated and gated by ``test_method_evidence.py``; release notes are history.
GENERATED_PAGES = re.compile(
    r"^docs/(technical-reference/method-evidence/(?!how-to-read)|development/release-notes-)"
)


def _match(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, p) for p in patterns)


def unmapped(files: Sequence[str], areas: Sequence[Area], internal: Mapping[str, str]) -> list[str]:
    """In-scope files that no area and no internal entry covers."""
    globs = [g for a in areas for g in a.code] + list(internal)
    return [f for f in files if _match(f, IN_SCOPE) and not _match(f, globs)]


def dead_globs(
    files: Sequence[str], areas: Sequence[Area], internal: Mapping[str, str]
) -> list[str]:
    """Globs that match no tracked file."""
    globs = [g for a in areas for g in a.code] + list(internal)
    return sorted({g for g in globs if not any(fnmatch.fnmatchcase(f, g) for f in files)})


def bad_pages(areas: Sequence[Area], standalone: Mapping[str, str], sources: set[str]) -> list[str]:
    """Pages the map names that are not documentation sources."""
    named = {p for a in areas for p in (*a.pages, *a.see_also)} | set(standalone)
    return sorted(named - sources)


def unreachable(
    areas: Sequence[Area], standalone: Mapping[str, str], reader: Iterable[str]
) -> list[str]:
    """Reader-facing pages no area names and no standalone entry explains."""
    named = {p for a in areas for p in (*a.pages, *a.see_also)} | set(standalone)
    return sorted(p for p in reader if p not in named and not GENERATED_PAGES.match(p))


def defined_names(source: str) -> set[str]:
    """Top-level functions, classes and assignments of a module."""
    names: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    # A dunder such as ``__version__`` is not a documented behaviour.
    return {n for n in names if not n.startswith("__")}


def exported(paths: Iterable[str], root: Path = ROOT) -> set[str]:
    """The names the ``__all__`` of each module in ``paths`` lists, in every form."""
    return set().union(*(exports((root / p).read_text(encoding="utf-8")) for p in paths))


def public_internals(
    internal: Mapping[str, str], files: Sequence[str], public: set[str]
) -> list[str]:
    """``INTERNAL`` modules that define a published name."""
    out = []
    for glob in internal:
        for path in (f for f in files if fnmatch.fnmatchcase(f, glob) and f.endswith(".py")):
            shared = defined_names((ROOT / path).read_text(encoding="utf-8")) & public
            if shared:
                out.append(f"{path}: {sorted(shared)}")
    return out


SOURCES = {p.relative_to(ROOT).as_posix() for p in READER_FACING}
READER = sorted(SOURCES)


def test_every_file_in_scope_is_mapped() -> None:
    assert unmapped(TRACKED, AREAS, INTERNAL) == []


def test_every_glob_matches_a_file() -> None:
    assert dead_globs(TRACKED, AREAS, INTERNAL) == []


def test_every_page_is_a_documentation_source() -> None:
    assert bad_pages(AREAS, STANDALONE_PAGES, SOURCES) == []


def test_every_reader_facing_page_is_reachable() -> None:
    assert unreachable(AREAS, STANDALONE_PAGES, READER) == []


def test_internal_entries_have_reasons_and_define_no_published_name() -> None:
    assert all(len(reason.split()) >= 2 for reason in INTERNAL.values())
    assert all(len(reason.split()) >= 2 for reason in STANDALONE_PAGES.values())
    assert public_internals(INTERNAL, TRACKED, exported(PUBLISHED)) == []


def test_published_lists_every_module_the_api_pages_publish() -> None:
    used: set[str] = set()
    for page in (ROOT / "docs" / "api").glob("*.*"):
        used |= set(re.findall(r"\bcleverly\.([a-z_]+)\b", page.read_text(encoding="utf-8")))
    modules = set()
    for name in used:
        if (ROOT / "src" / "cleverly" / name / "__init__.py").exists():
            modules.add(f"src/cleverly/{name}/__init__.py")
        elif (ROOT / "src" / "cleverly" / f"{name}.py").exists():
            modules.add(f"src/cleverly/{name}.py")
    assert modules - set(PUBLISHED) == set()
    assert all((ROOT / path).exists() for path in PUBLISHED)


def test_guards_exist() -> None:
    assert [g for a in AREAS for g in a.guards if not (ROOT / g).exists()] == []


def test_area_names_are_unique_and_pages_do_not_repeat_within_an_area() -> None:
    names = [a.name for a in AREAS]
    assert len(names) == len(set(names))
    for area in AREAS:
        assert len(set(area.pages)) == len(area.pages)
        assert not set(area.pages) & set(area.see_also), area.name


# ---- deliberate-mutation controls ------------------------------------------------------------


def test_a_new_module_with_no_area_fails() -> None:
    assert unmapped([*TRACKED, "src/cleverly/new_family.py"], AREAS, INTERNAL) == [
        "src/cleverly/new_family.py"
    ]


def test_a_glob_that_matches_nothing_fails() -> None:
    doctored = (*AREAS, Area("moved", ("src/cleverly/gone.py",), ("README.md",)))
    assert set(dead_globs(TRACKED, doctored, INTERNAL)) - set(
        dead_globs(TRACKED, AREAS, INTERNAL)
    ) == {"src/cleverly/gone.py"}


def test_a_missing_page_fails() -> None:
    doctored = (*AREAS, Area("typo", ("README.md",), ("docs/user-guide/no-such-page.md",)))
    assert bad_pages(doctored, STANDALONE_PAGES, SOURCES) == ["docs/user-guide/no-such-page.md"]


def test_an_orphan_page_fails() -> None:
    assert unreachable(AREAS, STANDALONE_PAGES, [*READER, "docs/user-guide/orphan.md"]) == [
        "docs/user-guide/orphan.md"
    ]


@pytest.mark.parametrize("glob", ["src/cleverly/study.py", "src/cleverly/methods.py"])
def test_hiding_a_public_module_in_internal_fails(glob: str) -> None:
    doctored = {**INTERNAL, glob: "not really internal"}
    assert public_internals(doctored, TRACKED, exported(PUBLISHED)) != []
