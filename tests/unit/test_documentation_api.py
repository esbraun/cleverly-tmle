"""The production site exposes its promised sections and the supported root API."""

from __future__ import annotations

import doctest
import importlib
import inspect
import pkgutil
import re
import tomllib
from collections import Counter
from functools import cache, cached_property
from types import ModuleType
from typing import Any

import pytest

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
SECTION = "^{name}\n-{{4,}}$"

#: The objects a reader calls to get from a question to a checked answer.  Examples and See
#: Also are required *here* and nowhere else, enforced by the two tests at the end of this
#: module rather than by ``numpydoc_validation_checks``, which can only require a check of
#: every object or of none.  A pro-forma example on all 140 curated objects would be noise,
#: and noise is what a reader learns to skip.
SPINE = (
    # Entry points and design.
    "cleverly.CausalStudy",
    "cleverly.PointTreatment",
    "cleverly.LongitudinalTreatment",
    "cleverly.IdentifiedEffect",
    # Estimands.
    "cleverly.ATE",
    "cleverly.ATT",
    "cleverly.ATC",
    "cleverly.CounterfactualMean",
    "cleverly.RiskRatio",
    "cleverly.RegimeContrast",
    "cleverly.ModifiedTreatmentPolicy",
    "cleverly.IncrementalEffect",
    "cleverly.ControlledDirectEffect",
    # Methods and configuration.
    "cleverly.EstimationMethod",
    "cleverly.TMLEMethod",
    "cleverly.CollaborativeTMLEMethod",
    "cleverly.DRTMLEMethod",
    "cleverly.ModelSpec",
    "cleverly.CrossFitting",
    "cleverly.Targeting",
    "cleverly.Inference",
    "cleverly.Runtime",
    # Results and assessment.
    "cleverly.estimators.TMLEResult",
    "cleverly.longitudinal.LongitudinalResult",
    "cleverly.ParameterEstimate",
    "cleverly.DiagnosticReport",
    "cleverly.ValidationReport",
    "cleverly.assessment.DiagnosticsFacade",
    "cleverly.assessment.SensitivityFacade",
    # Learners and data.
    "cleverly.SuperLearner",
    "cleverly.datasets.make_linear_ate",
    "cleverly.datasets.make_longitudinal",
)

#: Doctests that fit with the *default* learner library, which is a cross-fitted
#: :class:`~cleverly.learners.SuperLearner` over three candidates.  Each costs 30 to 120
#: seconds and the cost is the library rather than the sample size: shrinking ``n`` from
#: 1000 to 200 saves half a minute of a minute and a half, because the price is the number
#: of candidate fits.  They stay in the sweep and run under ``-m slow``.  Showing the
#: defaults is the whole point of these three, so making them cheap would mean documenting
#: a configuration no reader is told to use.
EXPENSIVE_DOCTESTS = frozenset(
    {
        "cleverly",
        "cleverly.estimators.tmle",
        "cleverly.longitudinal",
    }
)

#: A floor on what discovery is expected to find.  Well under the count at the time of
#: writing, because this is the one way a check over a discovered set fails open: a finder
#: that stopped matching would report success over an empty set, which reads exactly like
#: every example being fine.
_EXPECTED_DOCTESTS = 12


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


@cache
def _package_doctests() -> dict[str, doctest.DocTest]:
    """Return every docstring in ``cleverly`` that has at least one ``>>>``, keyed by object.

    Private modules are walked too.  A broken example is a broken example wherever it
    lives, and a module a reader is not routed to is one nobody has been reading.
    """
    finder = doctest.DocTestFinder()
    found: dict[str, doctest.DocTest] = {}
    modules = [cleverly, *_walk_submodules(cleverly)]
    for module in modules:
        for test in finder.find(module, module.__name__):
            if test.examples:
                found[test.name] = test
    return found


def _walk_submodules(package: ModuleType) -> list[ModuleType]:
    """Import every submodule so that discovery does not depend on what package import loads."""
    modules: list[ModuleType] = []
    for info in pkgutil.walk_packages(package.__path__, prefix=f"{package.__name__}."):
        modules.append(importlib.import_module(info.name))
    return modules


def _see_also_entries(docstring: str) -> list[tuple[str, str]]:
    """Return ``(reference, description)`` for each See Also entry, description possibly empty."""
    match = re.search(r"^See Also\n-{4,}\n(.*?)(?=\n\S|\Z)", docstring, re.MULTILINE | re.DOTALL)
    if match is None:
        return []
    entries: list[tuple[str, str]] = []
    for line in match.group(1).splitlines():
        if not line.strip() or line.startswith((" ", "\t")):
            continue
        reference, _, description = line.partition(":")
        entries.append((reference.strip(), description.strip()))
    return entries


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


def test_discovery_reaches_the_shipped_examples() -> None:
    """The sweep below is only as good as the set it runs over."""
    found = _package_doctests()
    assert found, "no doctests found under src/cleverly; check the walk in _package_doctests"
    assert len(found) >= _EXPECTED_DOCTESTS, (
        f"discovery found only {len(found)} doctests; check the walk in _package_doctests"
    )


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            name,
            marks=pytest.mark.slow if name in EXPENSIVE_DOCTESTS else (),
            id=name,
        )
        for name in sorted(_package_doctests())
    ],
)
def test_every_shipped_example_runs(name: str) -> None:
    """Every ``>>>`` in the package executes and prints what it says it prints.

    Discovered rather than listed.  An example is documentation a reader is invited to
    paste, so one that has drifted from the API is worse than no example: it fails in the
    reader's session and not in ours.  ``pytest --doctest-modules src/cleverly`` is the
    same sweep from outside, and the two are kept in step by running with the module
    globals the way that flag does.
    """
    test = _package_doctests()[name]
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    runner.run(test, out=lambda text: None)
    result = runner.summarize(verbose=False)
    assert result.failed == 0, f"{name}: {result.failed} of {result.attempted} examples failed"


@pytest.mark.parametrize("name", SPINE)
def test_every_spine_object_shows_the_reader_how_to_use_it(name: str) -> None:
    """A task-spine object carries a runnable example and a route to its neighbours.

    ``Examples`` answers "how do I call this", and ``See Also`` answers "what do I reach
    for instead".  Both entries of a See Also pair need a description, because a bare
    name tells a reader where to click and not why they would.
    """
    docstring = inspect.getdoc(_resolve(name)) or ""
    assert re.search(SECTION.format(name="Examples"), docstring, re.MULTILINE), (
        f"{name} is on the task spine and has no Examples section"
    )
    assert re.search(SECTION.format(name="See Also"), docstring, re.MULTILINE), (
        f"{name} is on the task spine and has no See Also section"
    )
    undescribed = [entry for entry in _see_also_entries(docstring) if not entry[1]]
    assert not undescribed, (
        f"{name}: See Also entries without a description: {[entry[0] for entry in undescribed]}"
    )


def test_the_task_spine_names_only_objects_the_index_publishes() -> None:
    """A spine entry that is not curated is a promise made about a retired object."""
    unknown = sorted(set(SPINE) - set(_object_names()))
    assert not unknown, f"SPINE names objects that docs/api/object-index.rst does not: {unknown}"
