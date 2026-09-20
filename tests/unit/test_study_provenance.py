"""A study manifest names every study-specific module its runner and properties import.

``study_module_sha256`` records the Python sources that produced a study's rows.  No test gates
those hashes against the working tree, so the one thing a reader can rely on is that the list is
*complete*: a helper that computed rows but is missing from the manifest leaves no record of the
bytes that ran.

The convention checked here: a registered study's manifest must list every ``tests`` module that
its runner module and its properties module import, directly or transitively within ``tests/``.
The shared evidence framework under ``tests/studies/evidence/`` and ``tests/parallel.py`` are
excluded, and the walk does not descend into them, because ``CLAUDE.md`` treats edits there as
free refactors.  The exclusion is a convention, not a claim that the framework never feeds a
result: ``tests/studies/evidence/pairing.py``, for one, pairs the rows a study publishes.
Package ``__init__.py`` files are not modules a study computes with and are not required either.
Imports are read statically with :mod:`ast`; no study module is imported to discover them.

A gap is accepted on one of two routes.  ``KNOWN_GAPS`` holds the gaps that predate this gate.
The ``study | source | judgement`` table in ``tests/canonical/provenance-revisions.md`` holds
gaps that a result-neutral refactor opened, such as a helper moved into a new module that
never ran.  Both routes are ratchets: a listed or declared module that the manifest records,
or that the study no longer imports, fails until its entry is removed.
"""

from __future__ import annotations

import ast
import functools
import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import pytest

from tests.documents import pipe_table
from tests.studies.evidence.registry import ROOT, StudyRecord, registered

STUDIES = registered()
IDS = [study.slug for study in STUDIES]

#: Shared machinery a study may import without naming it in its manifest.
FRAMEWORK_PREFIXES = ("tests/studies/evidence/",)
FRAMEWORK_FILES = frozenset({"tests/parallel.py"})

#: Ratchet: study slug -> modules its runner or properties import that its manifest omits.
#:
#: Most entries predate this gate.  An entry leaves when the ``StudyRecord`` and its manifest both
#: name the module: at the study's next regeneration, or by recording the hash of a file that git
#: shows unchanged since the run and declaring it in ``tests/canonical/provenance-revisions.md``.
#: Never add a hash of bytes that differ from the run.  The test fails on a gap that is not
#: listed here or declared in the ledger, and on a listed entry that is no longer a gap.
#:
#: Two routes open a gap, and they are not interchangeable.  A gap that a *result-neutral*
#: refactor opens goes in the ledger, because the study's committed rows still describe what the
#: code computes.  A gap that a *result-determining* change to a shared module opens cannot: the
#: ledger's judgement column must say ``result-neutral:`` and that would be false.  Such a gap is
#: listed here instead, with the regeneration that closes it named, and the entry leaves when the
#: study is regenerated.  Selective regeneration is the policy that makes this legal; an entry
#: that names no regeneration is the state this list exists to prevent.
KNOWN_GAPS: Mapping[str, frozenset[str]] = {
    "canonical-tmle": frozenset(
        {
            "tests/conftest.py",
            "tests/studies/canonical_cvtmle.py",
            "tests/studies/point_study_helpers.py",
        }
    ),
    "weighted-tmle": frozenset(
        {
            "tests/conftest.py",
        }
    ),
    "mar-tmle": frozenset(
        {
            "tests/conftest.py",
            "tests/studies/point_study_helpers.py",
        }
    ),
    "mar-drtmle": frozenset(
        {
            "tests/conftest.py",
            "tests/studies/point_study_helpers.py",
        }
    ),
    "point-msm": frozenset(
        {
            "tests/conftest.py",
        }
    ),
    "deterministic-regimes": frozenset(
        {
            "tests/conftest.py",
            "tests/studies/point_study_helpers.py",
        }
    ),
    "stochastic-regimes": frozenset(
        {
            "tests/conftest.py",
            "tests/studies/point_study_helpers.py",
        }
    ),
    "shift-policies": frozenset(
        {
            "tests/discrete_law.py",
            "tests/studies/point_study_helpers.py",
        }
    ),
    "incremental-interventions": frozenset(
        {
            "tests/conftest.py",
            "tests/studies/point_study_helpers.py",
        }
    ),
    "weighted-ltmle": frozenset(
        {
            "tests/studies/canonical_ltmle_crossfit.py",
            "tests/studies/ltmle_crossfit_properties.py",
        }
    ),
    "canonical-categorical-ltmle": frozenset(
        {
            "tests/studies/canonical_ltmle.py",
        }
    ),
    "longitudinal-msm": frozenset(
        {
            "tests/studies/canonical_ltmle.py",
            "tests/studies/ltmle_properties.py",
        }
    ),
    "canonical-ltmle-survival": frozenset(
        {
            "tests/discrete_law_longitudinal.py",
        }
    ),
    "canonical-ltmle-survival-crossfit": frozenset(
        {
            "tests/discrete_law_longitudinal.py",
        }
    ),
    "canonical-ltmle-competing": frozenset(
        {
            "tests/discrete_law_longitudinal.py",
        }
    ),
    "canonical-ltmle-competing-crossfit": frozenset(
        {
            "tests/discrete_law_longitudinal.py",
        }
    ),
}


#: Where a result-neutral refactor declares a module its study imports but its manifest omits.
REVISIONS = ROOT / "tests" / "canonical" / "provenance-revisions.md"
DECLARED_GAP_COLUMNS = ("study", "source", "judgement")


def declared_gaps(document: Path = REVISIONS) -> dict[str, frozenset[str]]:
    """``study slug -> modules`` from the ledger's ``study | source | judgement`` table."""
    declared: dict[str, set[str]] = {}
    for row in pipe_table(document, DECLARED_GAP_COLUMNS):
        slug, source = row["study"].strip("`"), row["source"].strip("`")
        assert source not in declared.get(slug, set()), f"{slug} declares {source} twice"
        assert row["judgement"].startswith("result-neutral:"), (
            f"{slug} declares {source} without a 'result-neutral: <reason>' judgement. A change "
            f"that can move the results is a regeneration, not a declaration"
        )
        declared.setdefault(slug, set()).add(source)
    return {slug: frozenset(sources) for slug, sources in declared.items()}


DECLARED_GAPS = declared_gaps()


def _is_framework(path: str) -> bool:
    return path in FRAMEWORK_FILES or path.startswith(FRAMEWORK_PREFIXES)


def _module_path(dotted: str, root: Path = ROOT) -> str | None:
    """The repository-relative source file of a ``tests`` module, or ``None`` for a package."""
    parts = dotted.split(".")
    if parts[0] != "tests":
        return None
    candidate = root.joinpath(*parts).with_suffix(".py")
    if candidate.is_file():
        return candidate.relative_to(root).as_posix()
    return None


def _imported_modules(path: str, root: Path = ROOT) -> set[str]:
    """Every ``tests`` module that the source file at ``path`` names in an import statement.

    Imports anywhere in the file count, including ones inside functions, because a study
    commonly defers an import to keep module import cheap.
    """
    source = root / path
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    package = Path(path).with_suffix("").parts[:-1]
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - (node.level - 1)]
                prefix = ".".join((*base, *(node.module.split(".") if node.module else ())))
            else:
                prefix = node.module or ""
            # ``from tests import discrete_law_mar`` imports a submodule, while
            # ``from tests.conftest import OracleOutcome`` imports a name from a module.
            names = [prefix, *(f"{prefix}.{alias.name}" for alias in node.names)]
        else:
            continue
        for name in names:
            resolved = _module_path(name, root)
            if resolved is not None:
                found.add(resolved)
    return found


def study_specific_imports(
    entry_points: Iterable[str],
    imports: Callable[[str], set[str]] = _imported_modules,
) -> frozenset[str]:
    """Entry points plus every non-framework ``tests`` module they reach transitively."""
    seen: set[str] = set()
    pending = [path for path in entry_points if not _is_framework(path)]
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        pending.extend(
            module for module in imports(path) if module not in seen and not _is_framework(module)
        )
    return frozenset(seen)


def provenance_findings(
    required: frozenset[str],
    recorded: frozenset[str],
    known: frozenset[str],
    declared: frozenset[str] = frozenset(),
) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """``(unaccepted gaps, stale KNOWN_GAPS entries, stale ledger declarations)``.

    All three empty passes.  ``known`` is the study's ``KNOWN_GAPS`` entry and ``declared`` is
    its ledger declarations; an entry or a declaration that is no longer a gap is stale.
    """
    gaps = required - recorded
    return gaps - known - declared, known - gaps, declared - gaps


def _required(study: StudyRecord) -> frozenset[str]:
    return _required_from(study.runner_module, study.properties_module)


@functools.cache
def _required_from(runner_module: str, properties_module: str) -> frozenset[str]:
    entry_points = [
        _module_path(runner_module),
        _module_path(properties_module),
    ]
    assert None not in entry_points, (
        f"{runner_module} or {properties_module} has no source file under tests/"
    )
    return study_specific_imports(path for path in entry_points if path is not None)


def _recorded(study: StudyRecord) -> frozenset[str]:
    manifest = json.loads(study.artifact("manifest.json").read_text(encoding="utf-8"))
    return frozenset(manifest["study_module_sha256"])


@pytest.mark.parametrize("study", STUDIES, ids=IDS)
def test_the_manifest_names_every_study_specific_module(study: StudyRecord) -> None:
    unlisted, fixed, stale = provenance_findings(
        _required(study),
        _recorded(study),
        KNOWN_GAPS.get(study.slug, frozenset()),
        DECLARED_GAPS.get(study.slug, frozenset()),
    )
    assert not unlisted, (
        f"{study.slug} imports {sorted(unlisted)} but its manifest records no hash for them. "
        f"If the change can move the study's results, add the modules to the StudyRecord "
        f"modules field and regenerate the study. If the change is result-neutral, declare "
        f"each module in the study | source | judgement table of {REVISIONS.name}"
    )
    assert not fixed, (
        f"{study.slug} no longer omits {sorted(fixed)}. Remove them from KNOWN_GAPS so the "
        f"ratchet keeps the gain"
    )
    assert not stale, (
        f"{study.slug} no longer omits {sorted(stale)}. Remove their declarations from "
        f"{REVISIONS.name} so the ledger explains only real gaps"
    )


def test_every_allowlisted_gap_names_a_registered_study() -> None:
    assert set(KNOWN_GAPS) <= set(IDS), sorted(set(KNOWN_GAPS) - set(IDS))
    assert all(KNOWN_GAPS.values()), "an empty KNOWN_GAPS entry explains nothing; remove it"
    assert set(DECLARED_GAPS) <= set(IDS), sorted(set(DECLARED_GAPS) - set(IDS))
    for slug, declared in DECLARED_GAPS.items():
        assert not declared & KNOWN_GAPS.get(slug, frozenset()), (
            f"{slug} lists {sorted(declared & KNOWN_GAPS[slug])} in KNOWN_GAPS and in the "
            f"ledger; one gap has one reason"
        )


def test_the_natural_course_study_reaches_its_shared_helpers() -> None:
    """A hand-checked reach, so a resolver that finds nothing cannot pass every study."""
    (study,) = (study for study in STUDIES if study.slug == "mar-natural-course-tmle")
    required = _required(study)
    assert {
        "tests/studies/canonical_mar_natural_course.py",
        "tests/studies/mar_natural_course_properties.py",
        "tests/studies/missing_outcome_study_helpers.py",
        "tests/studies/point_study_helpers.py",
        "tests/discrete_law_mar.py",
        "tests/conftest.py",
    } <= required
    assert not any(_is_framework(module) for module in required)


def test_a_module_dropped_from_a_manifest_is_reported() -> None:
    """Mutation control: remove one recorded module from an in-memory copy of a manifest."""
    (study,) = (study for study in STUDIES if study.slug == "stacked-mar-natural-course-cvtmle")
    required = _required(study)
    recorded = _recorded(study)
    known = KNOWN_GAPS.get(study.slug, frozenset())
    declared = DECLARED_GAPS.get(study.slug, frozenset())
    clean = (frozenset(), frozenset(), frozenset())
    assert provenance_findings(required, recorded, known, declared) == clean
    dropped = "tests/studies/point_study_helpers.py"
    assert dropped in recorded
    unlisted, fixed, stale = provenance_findings(required, recorded - {dropped}, known, declared)
    assert unlisted == {dropped}
    assert not fixed
    assert not stale


def test_a_repaired_allowlisted_gap_is_reported() -> None:
    required = frozenset({"tests/a.py", "tests/b.py"})
    unlisted, fixed, stale = provenance_findings(
        required, frozenset({"tests/a.py", "tests/b.py"}), frozenset({"tests/b.py"})
    )
    assert unlisted == frozenset()
    assert fixed == {"tests/b.py"}
    assert stale == frozenset()


def test_a_ledger_declaration_accepts_a_refactored_module_and_ratchets(tmp_path: Path) -> None:
    """A helper moved into a new module passes once declared, and fails once the gap closes."""
    ledger = tmp_path / "provenance-revisions.md"
    ledger.write_text(
        "# Provenance revisions\n\n"
        "| source | recorded | current | judgement |\n"
        "| --- | --- | --- | --- |\n\n"
        "| study | source | judgement |\n"
        "| --- | --- | --- |\n"
        "| `some-study` | `tests/studies/x_helpers.py` | result-neutral: moved, not changed |\n",
        encoding="utf-8",
    )
    declared = declared_gaps(ledger)
    assert declared == {"some-study": {"tests/studies/x_helpers.py"}}

    required = frozenset({"tests/studies/runner.py", "tests/studies/x_helpers.py"})
    recorded = frozenset({"tests/studies/runner.py"})
    assert provenance_findings(required, recorded, frozenset()) == (
        {"tests/studies/x_helpers.py"},
        frozenset(),
        frozenset(),
    )
    assert provenance_findings(required, recorded, frozenset(), declared["some-study"]) == (
        frozenset(),
        frozenset(),
        frozenset(),
    )
    # Regenerated: the manifest now records the module, so the declaration is stale.
    assert provenance_findings(required, required, frozenset(), declared["some-study"]) == (
        frozenset(),
        frozenset(),
        {"tests/studies/x_helpers.py"},
    )


def test_a_ledger_declaration_without_a_result_neutral_judgement_is_refused(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "provenance-revisions.md"
    ledger.write_text(
        "| study | source | judgement |\n"
        "| --- | --- | --- |\n"
        "| `some-study` | `tests/studies/x_helpers.py` | moved |\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="result-neutral"):
        declared_gaps(ledger)


def test_the_walk_is_transitive_and_stops_at_the_framework(tmp_path: Path) -> None:
    files = {
        "tests/studies/runner.py": (
            "from tests.studies import helper\n"
            "from tests.studies.evidence.registry import StudyRecord\n"
            "from tests.parallel import STUDY_JOBS\n"
        ),
        "tests/studies/helper.py": "def f():\n    from .deep import g\n",
        "tests/studies/deep.py": "from tests import law as mar\n",
        "tests/law.py": "",
        "tests/parallel.py": "from tests import hidden\n",
        "tests/hidden.py": "",
        "tests/studies/evidence/registry.py": "from tests import hidden\n",
    }
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text, encoding="utf-8")

    reached = study_specific_imports(
        ["tests/studies/runner.py"], lambda path: _imported_modules(path, tmp_path)
    )

    assert reached == {
        "tests/studies/runner.py",
        "tests/studies/helper.py",
        "tests/studies/deep.py",
        "tests/law.py",
    }
