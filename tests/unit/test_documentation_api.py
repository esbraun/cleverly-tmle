"""The production site exposes its promised sections and the supported root API."""

from __future__ import annotations

import doctest
import importlib
import inspect
import re
import tomllib
from collections import Counter
from functools import cached_property
from types import ModuleType
from typing import Any

import cleverly
import cleverly.datasets
from tests.documents import ROOT

SITE_INDEX = ROOT / "docs" / "index.md"
API_ROOT = ROOT / "docs" / "api"
README = ROOT / "README.md"
SPHINX_CONFIG = ROOT / "docs" / "conf.py"
PAGES_WORKFLOW = ROOT / ".github" / "workflows" / "pages.yml"
PUBLIC_DOCS_URL = "https://esbraun.github.io/cleverly-tmle/"
OBJECT_PATTERN = re.compile(r"^\s*(cleverly\.[A-Za-z0-9_.]+)\s*$", re.MULTILINE)
TASK_EXAMPLES = (
    "cleverly.CausalStudy",
    "cleverly.PointTreatment",
    "cleverly.ATE",
    "cleverly.ModelSpec",
    "cleverly.TMLEMethod",
    "cleverly.datasets.make_linear_ate",
)


def _object_names() -> tuple[str, ...]:
    """Return the ordered public contract from the autosummary manifest."""
    source = (API_ROOT / "object-index.rst").read_text(encoding="utf-8")
    return tuple(OBJECT_PATTERN.findall(source))


def _resolve(name: str) -> Any:
    """Import an object even when its leaf module is not loaded at package import."""
    parts = name.split(".")
    value: Any = importlib.import_module(parts[0])
    try:
        for part in parts[1:]:
            value = getattr(value, part)
    except AttributeError:
        pass
    else:
        return value

    for boundary in range(len(parts), 0, -1):
        try:
            value = importlib.import_module(".".join(parts[:boundary]))
        except ModuleNotFoundError:
            continue
        for part in parts[boundary:]:
            value = getattr(value, part)
        return value
    raise AssertionError(f"cannot import public API object {name}")


def _documented_callable(member: Any) -> Any | None:
    """Unwrap a direct class member when users can call or read it."""
    if isinstance(member, (classmethod, staticmethod)):
        return member.__func__
    if isinstance(member, property):
        return member.fget
    if isinstance(member, cached_property):
        return member.func
    if inspect.isfunction(member):
        return member
    return None


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


def test_the_readme_routes_to_the_canonical_site_immediately() -> None:
    """The repository landing page sends readers to the rendered docs before its first section."""
    text = README.read_text(encoding="utf-8")
    opening = text.partition("\n## ")[0]
    assert PUBLIC_DOCS_URL in opening


def test_publication_metadata_agrees_on_one_canonical_site() -> None:
    """Sphinx and package metadata cannot advertise different official documentation sites."""
    config = SPHINX_CONFIG.read_text(encoding="utf-8")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    urls = project["project"]["urls"]

    assert f'html_baseurl = "{PUBLIC_DOCS_URL}"' in config
    assert urls["Homepage"] == PUBLIC_DOCS_URL
    assert urls["Documentation"] == PUBLIC_DOCS_URL


def test_pages_workflow_builds_and_deploys_the_sphinx_site() -> None:
    """The canonical site is produced by the warning-as-error build, not a second toolchain."""
    workflow = PAGES_WORKFLOW.read_text(encoding="utf-8")
    required = {
        "sphinx-build -W --keep-going -b html docs docs/_build/html",
        "actions/configure-pages@v5",
        "actions/upload-pages-artifact@v4",
        "actions/deploy-pages@v4",
        "pages: write",
        "id-token: write",
    }
    missing = sorted(item for item in required if item not in workflow)
    assert not missing, f"pages workflow is missing its publication contract: {missing}"


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

    **This is load-bearing and not tidiness.**  ``object-index.rst`` is what generates every
    object page.  It is the only file whose ``.. autosummary::`` carries ``:toctree:``, and
    ``sphinx.ext.autosummary`` writes a stub only for an entry that has one.  The nine
    categorised ``.md`` pages carry the same 140 names without a toctree, so they render tables
    and generate nothing.  They wrap the directive in an ``eval-rst`` block because MyST parses a
    ``{autosummary}`` fence body as Markdown, which leaves the ``:py:obj:`` name cell unresolved
    and the first column of every table empty.

    An object added to a ``.md`` page alone therefore links to a stub that was never written, and
    the warning-as-error build fails somewhere unrelated-looking.  One added to the ``.rst`` alone
    generates a page no categorised table lists.  Naming both halves here turns either mistake into
    a message that says which file is short.
    """
    category_names = [
        name
        for path in sorted(API_ROOT.glob("*.md"))
        for name in OBJECT_PATTERN.findall(path.read_text(encoding="utf-8"))
    ]
    categorised = set(category_names)
    indexed = set(_object_names())

    assert categorised, "no objects found on the categorised API pages; check the pattern"
    assert categorised == indexed, (
        f"docs/api/*.md and docs/api/object-index.rst disagree. "
        f"Only on a categorised page (no stub will be generated): {sorted(categorised - indexed)}. "
        f"Only in the object index (no table lists it): {sorted(indexed - categorised)}"
    )

    duplicate_categories = {
        name: count for name, count in Counter(category_names).items() if count != 1
    }
    assert not duplicate_categories, (
        f"each public API object needs one category; repeated entries: {duplicate_categories}"
    )


def test_the_object_index_is_an_importable_documented_public_contract() -> None:
    """Every curated object imports and supplies its own user-facing summary."""
    names = _object_names()
    assert len(names) == len(set(names)), "object-index.rst contains duplicate API objects"

    missing: list[str] = []
    for name in names:
        value = _resolve(name)
        assert not isinstance(value, ModuleType), f"{name} resolves to a module, not an API object"
        if not (value.__doc__ or "").strip():
            missing.append(name)

    assert not missing, f"public API objects without direct docstrings: {missing}"


def test_public_classes_document_each_direct_public_member() -> None:
    """Generated class pages must not expose unexplained methods or properties."""
    missing: list[str] = []
    for object_name in _object_names():
        value = _resolve(object_name)
        if not inspect.isclass(value):
            continue
        for member_name, raw_member in vars(value).items():
            if member_name.startswith("_"):
                continue
            member = _documented_callable(raw_member)
            if member is not None and not (member.__doc__ or "").strip():
                missing.append(f"{object_name}.{member_name}")

    assert not missing, f"public class members without direct docstrings: {missing}"


def test_task_level_api_examples_are_present_and_runnable() -> None:
    """Core workflow examples execute without fitting a slow statistical study."""
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    missing_sections: list[str] = []

    for name in TASK_EXAMPLES:
        value = _resolve(name)
        docstring = inspect.getdoc(value) or ""
        if not re.search(r"^Examples\n-{8,}$", docstring, re.MULTILINE):
            missing_sections.append(name)
            continue
        test = doctest.DocTestParser().get_doctest(docstring, {}, name, name, 0)
        runner.run(test)

    assert not missing_sections, f"core API objects without Examples sections: {missing_sections}"
    failures, _ = runner.summarize(verbose=False)
    assert failures == 0
