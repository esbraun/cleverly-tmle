"""Contract item 8: every summary-bearing class has a fixture, and every note code fires.

``tests/unit/test_summary_contract.py`` runs each contract item over the fixtures of
``tests/unit/summary_fixtures``. These tests check that the fixtures reach everything a
summary can print. A class with a ``summary``, a ``to_frame`` or a ``to_dict`` that no
fixture builds is a printout nothing checks. A note code that no fixture fires is a text
that escapes the house-style and uniqueness checks.

The deliberate-mutation controls remove the fixtures that reach the last uncovered
objects and codes, and show that the checks then fail.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import typing
from typing import Any

import pytest

import cleverly
from cleverly._summary import vocabulary
from tests.unit.summary_fixtures import setup as setup_fixtures
from tests.unit.test_summary_contract import FIXTURES, _fitted

#: Objects X89 plan 3.1 marks "Summary": ``summary()`` returns the shared summary, so a
#: fixture builds each one.
SUMMARY_CLASSES = (
    "CausalStudy",
    "IdentifiedEffect",
    "TMLEResult",
    "TMLEResultSet",
    "ParameterEstimate",
    "CVTargeting",
    "CTMLESelection",
    "VariableImportanceResult",
    "LongitudinalResult",
    "StructuralNestedMeanResult",
    "StructuralModelCheck",
    "GFormulaReport",
    "SupportReport",
    "PolicySupport",
    "IncrementalSupport",
    "RefutationResult",
    "StudyResult",
    "PhaseProfile",
    "StudyProtocol",
    "Replayability",
)

#: Objects plan 3.1 marks "Repr": a one-line repr and a notebook display, reached through a
#: parent whose summary shows them. ``REPR_CLASSES`` that no fixture builds are built here
#: from the parent that holds them.
REPR_CLASSES = (
    "MethodAvailability",
    "CTMLEDerivativeFit",
    "CTMLEOutcomeAdaptiveFit",
    "AdaptiveIPTWFit",
    "ReducedFit",
    "SuperLearnerDiagnostics",
    "MSMRegimenFit",
    "StructuralDiagnostics",
    "RegimeSupport",
    "NuisanceModelReport",
    "AssessmentItem",
    "AssessmentCapability",
    "RefutationTest",
    "BootstrapSummary",
    "EstimandSummary",
    "CausalData",
    "LongitudinalData",
    "SplitPlan",
    "CrossFitPlan",
    "Provenance",
    "DiagnosticsFacade",
    "SensitivityFacade",
)

#: Classes with a ``summary``, ``to_frame`` or ``to_dict`` that no fixture can build, and why.
EXEMPT = {
    "CausalResult": "a Protocol; each class that implements it has fixtures",
}


def _public_classes() -> dict[str, type]:
    """Every public class defined in the package, by name."""
    found: dict[str, type] = {}
    for info in pkgutil.walk_packages(cleverly.__path__, "cleverly."):
        module = importlib.import_module(info.name)
        for name, value in vars(module).items():
            if (
                inspect.isclass(value)
                and value.__module__ == info.name
                and not name.startswith("_")
                and value.__qualname__ == name
            ):
                found.setdefault(name, value)
    return found


CLASSES = _public_classes()


def _bearing() -> set[str]:
    """The classes that print or convert themselves: plan 3.1's lists and the introspected."""
    introspected = {
        name
        for name, cls in CLASSES.items()
        if any(hasattr(cls, method) for method in ("summary", "to_frame", "to_dict"))
    }
    return introspected | set(SUMMARY_CLASSES) | set(REPR_CLASSES)


def _built(keys: typing.Iterable[str]) -> set[str]:
    """The names of the classes the fixtures ``keys`` build, with each base class."""
    names: set[str] = set()
    for key in keys:
        names.update(cls.__name__ for cls in type(_fitted(key)).__mro__)
    return names


def _repr_only() -> dict[str, Any]:
    """An instance of each Repr class that no fixture builds, from the parent that holds it."""
    from tests.unit.summary_fixtures import point_tmle

    fit = _fitted("point_tmle:canonical")
    study = setup_fixtures.point_study()
    return {
        "MethodAvailability": setup_fixtures.ate_methods()[0],
        "NuisanceModelReport": _fitted("models:models_cross_fitted").models[0],
        "AssessmentItem": _fitted("battery:battery_every_status").diagnostics.items[0],
        "AssessmentCapability": fit.diagnostics.capabilities[0],
        "RefutationTest": _fitted("checks:refute_default").tests[0],
        "EstimandSummary": next(iter(_fitted("setup:simulation_passing").summaries.values())),
        "CausalData": study.data,
        "LongitudinalData": setup_fixtures.longitudinal_study().data,
        "SplitPlan": fit.split_plan,
        "CrossFitPlan": fit.config.crossfit,
        "Provenance": fit.provenance,
        "DiagnosticsFacade": point_tmle.single_ate().diagnostics,
        "SensitivityFacade": point_tmle.single_ate().sensitivity,
    }


def _uncovered(keys: typing.Iterable[str]) -> list[str]:
    built = _built(keys)
    parentless = set(_repr_only())
    return sorted(
        name
        for name in _bearing()
        if name in CLASSES and name not in built | parentless | set(EXEMPT)
    )


def _unfired(keys: typing.Iterable[str]) -> list[str]:
    fired: set[str] = set()
    for key in keys:
        for detail in ("short", "full"):
            fired.update(_fitted(key).summary(detail=detail).note_codes)
    return sorted(set(vocabulary.NOTE_CODES) - fired)


KEYS = sorted(FIXTURES)


def test_every_marked_class_exists() -> None:
    """A rename that drops a class from plan 3.1's lists fails here, not by going unchecked."""
    missing = [name for name in (*SUMMARY_CLASSES, *REPR_CLASSES) if name not in CLASSES]
    assert not missing, missing


def test_every_class_with_a_summary_or_a_frame_has_a_fixture() -> None:
    assert _uncovered(KEYS) == [], (
        "classes with a summary, a to_frame or a to_dict that no fixture builds; add a "
        "fixture module entry that builds each one, or list the class in EXEMPT with the reason"
    )


def test_every_summary_class_is_built_by_a_fixture_rather_than_a_parent() -> None:
    """A Summary class prints its own summary, so a parent's fixture does not stand in for it."""
    built = _built(KEYS)
    assert [name for name in SUMMARY_CLASSES if name not in built] == []


def test_every_repr_only_object_prints_one_short_line() -> None:
    for name, obj in _repr_only().items():
        assert type(obj).__name__ == name, (name, type(obj).__name__)
        text = repr(obj)
        assert "\n" not in text and len(text) <= 100, (name, text)


def test_every_note_code_of_the_vocabulary_is_fired_by_a_fixture() -> None:
    assert _unfired(KEYS) == [], (
        "note codes no fixture fires, so their texts escape the house-style and uniqueness "
        "checks of the contract; add a fixture that fires each, or remove the unused code"
    )


def test_dropping_the_coverage_fixtures_leaves_notes_and_classes_unreached() -> None:
    """Mutation control: without ``summary_fixtures/coverage.py`` both checks fail.

    The positive witness that the checks above read the fixtures: take away the module that
    fires the last codes and builds the last classes, and each check reports them.
    """
    without = [key for key in KEYS if not key.startswith("coverage:")]
    unfired = _unfired(without)
    assert {"no_method", "why_not", "isotonic_moves", "strength_clipped"} <= set(unfired)
    assert "as_treated_missing" in unfired
    with pytest.raises(AssertionError):
        assert _unfired(without) == []


def test_a_class_nothing_builds_is_reported() -> None:
    """Mutation control: a fixture list that holds no result leaves its class uncovered."""
    only_setup = [key for key in KEYS if key.startswith("setup:")]
    uncovered = _uncovered(only_setup)
    assert "TMLEResult" in uncovered and "LongitudinalResult" in uncovered
