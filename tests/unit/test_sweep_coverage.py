"""Every estimator the library ships is a kind of fit that the capability sweep asks.

The sweep in :mod:`tests.unit.test_capability_row_sweep` asks each refutation, validation and
sensitivity row about each kind of fit in :data:`~tests.unit._capability_sweep_support.KINDS`.
A kind that is not there is never asked, so a new estimator would inherit its family's rows with
no test of whether they run on it. This module reads from the code every public fit result
class, fitted method, support engine, target family, registered point estimand, and each value
of the support key's outcome, missing-data, treatment, weighting, splitting and dependence axes.
It fails when no kind fits one of them, unless :data:`UNSWEPT` names it with a reason and the
row that adds its kind.

:data:`UNSWEPT` is pinned both ways. An entry whose element some kind now fits fails, and so does
an entry whose element the code no longer has. The row that adds the kind deletes the entry.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import pkgutil
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import pytest

import cleverly
from cleverly import _support
from cleverly._support import REGIMEN_KINDS, SupportKey, registered_rows
from cleverly._support_rows import FAMILY_THEORY
from cleverly.assessment import ASSESSMENT_CAPABILITIES
from cleverly.targets import TARGETS
from tests.unit._capability_sweep_support import KINDS, Kind


@dataclass(frozen=True)
class Exemption:
    """Why no kind of the sweep fits one element yet, and which row adds it.

    Parameters
    ----------
    reason : str
        What is missing.
    owner : str
        The row that adds the kind and deletes this entry, as a plans-repo row id.
    """

    reason: str
    owner: str


#: A plans-repo row id: a roadmap item, a future investigation, a fix row, or a P or R row.
OWNER = re.compile(r"(X|F|P|R)\d+|Fix-\d+")

#: The support key axes whose every registered value needs a kind. ``qualifiers`` is left out:
#: it records study-cell details (a time window, a cluster layout) that change no capability row.
OPTION_AXES = ("outcome", "missing", "treatment", "weighting", "splitting", "dependence")

#: Public classes named ``...Result`` that an assessment returns, not a fit. Each must still
#: exist, so a renamed output cannot leave a stale name here.
ASSESSMENT_OUTPUTS = frozenset(
    {
        "BenchmarkResult",
        "BootstrapResult",
        "RefutationResult",
        "SimulatedConfoundingResult",
        "StudyResult",
    }
)


def _exempt(reason: str, owner: str, names: Iterable[str]) -> dict[str, Exemption]:
    return {name: Exemption(reason, owner) for name in names}


#: Each element no kind fits, by its tagged name (see :func:`covered`).
UNSWEPT: Mapping[str, Exemption] = {
    **_exempt(
        "a fit result class with no assessment facade: it declares no assessment_family, no "
        "diagnostics or sensitivity facade and no assess(), so it has no capability row to ask",
        "X104",
        ("result:DoseResponseResult", "result:VariableImportanceResult"),
    ),
    **_exempt(
        "the effect curve's assessment surface (CurveDiagnostics) has no sensitivity facade, no "
        "combined report (assess) and no per-request row resolution, so the sweep's instrument "
        "cannot ask it; tests/unit/test_curve_assessment.py checks its rows directly",
        "X104",
        (
            "result:EffectCurveResult",
            "method:effect_modification/ep",
            "method:effect_modification/tmle",
        ),
    ),
    **_exempt(
        "the structural nested mean result has a sensitivity facade but no diagnostics facade "
        "(its diagnostics field is the fit-time StructuralDiagnostics report) and no assess(), "
        "so the sweep has no facade call to ask (the free refute() does run on it)",
        "X104",
        (
            "result:StructuralNestedMeanResult",
            "method:structural/snmm",
            "engine:snmm",
            "family:blips",
        ),
    ),
}


def _engine(engine: tuple[Any, ...]) -> str:
    return "engine:" + "/".join(str(part) for part in engine)


def _key_values(key: SupportKey) -> set[str]:
    axes = key.axes()
    found = {_engine(key.engine), *(f"family:{family}" for family in key.family)}
    for axis in OPTION_AXES:
        value = axes[axis]
        found.update(f"{axis}:{part}" for part in (value if isinstance(value, tuple) else (value,)))
    return found


@functools.cache
def result_classes() -> frozenset[str]:
    """The public fit result classes: every public ``...Result`` class but a protocol or an output.

    Returns
    -------
    frozenset of str
        Class names, read from the ``__all__`` of every public module of :mod:`cleverly`.
    """
    found: set[str] = set()
    for module in pkgutil.walk_packages(cleverly.__path__, "cleverly."):
        if any(part.startswith("_") for part in module.name.split(".")[1:]):
            continue
        namespace = importlib.import_module(module.name)
        for name in getattr(namespace, "__all__", ()):
            value = getattr(namespace, name, None)
            if (
                inspect.isclass(value)
                and name.endswith("Result")
                and not getattr(value, "_is_protocol", False)
            ):
                found.add(name)
    return frozenset(found)


def universe() -> frozenset[str]:
    """Every element the sweep must fit, read from the code that declares it.

    Returns
    -------
    frozenset of str
        Tagged names: ``result:<class>`` for each public fit result class,
        ``method:<assessment family>/<fitted method>`` from the capability rows,
        ``engine:<engine>`` and ``<axis>:<value>`` from the registered support rows,
        ``family:<id>`` from the target family maps, the theory anchors, the regimen kinds and
        the support rows, and ``estimand:<name>`` from the registered point targets.
    """
    rows = registered_rows()
    families = {
        *FAMILY_THEORY,
        *_support._ARM_FAMILIES.values(),
        *_support._AXIS_FAMILIES.values(),
        _support.OPTIMAL_RULE_FAMILY,
        _support.LOG_CRR_PROJECTION_FAMILY,
        *REGIMEN_KINDS,
    }
    return frozenset(
        {
            *(f"result:{name}" for name in result_classes() - ASSESSMENT_OUTPUTS),
            *(
                f"method:{row.result_family}/{method}"
                for row in ASSESSMENT_CAPABILITIES
                for method in row.methods
            ),
            *(f"family:{family}" for family in families),
            *(value for row in rows for value in _key_values(row.key)),
            *(f"estimand:{name}" for name in TARGETS),
        }
    )


def covered(result: Any) -> frozenset[str]:
    """The tagged elements one fitted result stands for, read from the result itself.

    Parameters
    ----------
    result : object
        A fitted result of a kind.

    Returns
    -------
    frozenset of str
        Its class and method, and, where its configuration carries them, its support key's
        engine, target families and axis values, and its requested point estimands.
    """
    found = {
        f"result:{type(result).__name__}",
        f"method:{result.assessment_family}/{result.fitted_method}",
    }
    config = getattr(result, "config", None)
    key = getattr(config, "support_key", None)
    if key is not None:
        # A kind may hold a key value no registered row names (a supplied split plan); only
        # the universe's elements are required, so the extra value is harmless.
        found |= _key_values(key)
    found.update(f"estimand:{name}" for name in getattr(config, "estimands", None) or ())
    return frozenset(found)


def swept(kinds: Mapping[str, Kind]) -> frozenset[str]:
    """Every element some kind of ``kinds`` fits."""
    return frozenset().union(*(covered(kind.build()) for kind in kinds.values()))


@functools.cache
def _swept_kinds() -> frozenset[str]:
    return swept(KINDS)


def gaps(
    elements: frozenset[str], fitted: frozenset[str], exempt: Mapping[str, Exemption]
) -> list[str]:
    """Every way the elements, the kinds and the exemptions disagree. Empty is agreement."""
    found = [
        f"{name}: no kind of the sweep fits it" for name in sorted(elements - fitted - set(exempt))
    ]
    found += [
        f"{name}: exempt, and no longer in the code" for name in sorted(set(exempt) - elements)
    ]
    found += [f"{name}: exempt, and a kind now fits it" for name in sorted(set(exempt) & fitted)]
    # A method no capability row declares has no row for the sweep to ask about.
    found += [
        f"{name}: a kind fits it, and no capability row declares it"
        for name in sorted(fitted - elements)
        if name.startswith("method:")
    ]
    return found


def test_every_estimator_is_a_kind_of_the_sweep_or_an_owned_exemption() -> None:
    assert gaps(universe(), _swept_kinds(), UNSWEPT) == []


def test_every_exemption_names_a_reason_and_an_owning_row() -> None:
    for name, exemption in UNSWEPT.items():
        assert exemption.reason.strip(), name
        assert OWNER.fullmatch(exemption.owner), (name, exemption.owner)


def test_every_assessment_output_is_a_public_result_class() -> None:
    assert result_classes() >= ASSESSMENT_OUTPUTS


def test_the_universe_reads_every_axis() -> None:
    """A universe that lost an axis would cover nothing on it and pass."""
    elements = universe()
    for tag in ("result", "method", "engine", "family", "estimand", *OPTION_AXES):
        assert any(name.startswith(f"{tag}:") for name in elements), tag
    assert {"result:TMLEResult", "method:point/tmle", "family:P1", "estimand:ate"} <= elements
    assert "result:CausalResult" not in elements  # a protocol, not a fit


def test_a_removed_kind_fails_on_what_only_it_fits() -> None:
    """The live control: drop every adaptive-IPTW kind and rebuild the coverage from the rest."""
    rest = {name: kind for name, kind in KINDS.items() if not name.startswith("iptw")}
    found = gaps(universe(), swept(rest), UNSWEPT)
    assert any(line.startswith("method:point/adaptive_iptw:") for line in found), found
    assert any(line.startswith("engine:adaptive_iptw:") for line in found), found


@pytest.mark.parametrize(
    ("elements", "exempt", "element"),
    [
        pytest.param({"engine:new_engine"}, {}, "engine:new_engine", id="new engine"),
        pytest.param({"result:NewResult"}, {}, "result:NewResult", id="new result class"),
        pytest.param(
            set(),
            {"method:point/tmle": Exemption("stale", "Fix-15")},
            "method:point/tmle",
            id="exemption outlives its gap",
        ),
        pytest.param(
            set(),
            {"family:P99": Exemption("names nothing", "Fix-15")},
            "family:P99",
            id="exemption names nothing",
        ),
    ],
)
def test_each_control_makes_the_check_fail(
    elements: set[str], exempt: dict[str, Exemption], element: str
) -> None:
    found = gaps(universe() | elements, _swept_kinds(), {**UNSWEPT, **exempt})
    assert any(line.startswith(f"{element}:") for line in found), found


def test_a_method_no_row_declares_fails() -> None:
    found = gaps(universe(), _swept_kinds() | {"method:point/unknown"}, UNSWEPT)
    assert any(line.startswith("method:point/unknown:") for line in found), found
