"""Every estimator the library ships is a kind of fit that the capability sweep asks.

The sweep in :mod:`tests.unit.test_capability_row_sweep` asks each refutation, validation and
sensitivity row about each kind of fit in :data:`~tests.unit._capability_sweep_support.KINDS`.
A kind that is not there is never asked, so a new estimator would inherit its family's rows with
no test of whether they run on it. This module reads, from the code, every fitted method, support
engine, target family and registered point estimand. It fails when no kind fits one of them,
unless :data:`UNSWEPT` names it with a reason and an owner.

:data:`UNSWEPT` is pinned both ways. An entry whose element some kind now fits fails, and so does
an entry whose element the code no longer has. The row that adds the kind deletes the entry.

Plots are covered by the plot result matrix of ``tests/unit/test_plotting_result_matrix.py``.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import pytest

from cleverly import _support
from cleverly._support import REGIMEN_KINDS, registered_rows
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
        The row that adds the kind and deletes this entry.
    """

    reason: str
    owner: str


_KINDS_MISSING = "no kind of the capability sweep fits it yet"
_FIX15 = "Fix-15"

#: Each element no kind fits, by its tagged name (see :func:`covered`).
UNSWEPT: Mapping[str, Exemption] = {
    name: Exemption(_KINDS_MISSING, _FIX15)
    for name in (
        "method:augmented_trial/atmle",
        "method:effect_modification/ep",
        "method:effect_modification/tmle",
        "method:longitudinal/sdr",
        "method:structural/snmm",
        "engine:collaborative_tmle/derivative/None",
        "engine:collaborative_tmle/discrete/None",
        "engine:collaborative_tmle/greedy",
        "engine:collaborative_tmle/greedy/None",
        "engine:collaborative_tmle/lasso/None",
        "engine:collaborative_tmle/learners/None",
        "engine:collaborative_tmle/oat/per_arm/parametric_stack",
        "engine:drtmle/univariate",
        "engine:sdr",
        "engine:snmm",
        "family:P8",
        "family:P12",
        "family:P13",
        "family:P14",
        "family:blips",
        "family:competing",
        "family:learned",
        "family:msm",
        "family:mtp",
        "family:repeated",
        "family:separable",
        "family:stochastic",
        "estimand:ate_learned_rule",
        "estimand:ate_rr_tilt",
        "estimand:cate_projection",
        "estimand:ey_known_regime",
        "estimand:ey_optimal_rule",
        "estimand:ey_rr_tilt",
        "estimand:log_crr_projection",
    )
}


def _engine(engine: tuple[Any, ...]) -> str:
    return "engine:" + "/".join(str(part) for part in engine)


def universe() -> frozenset[str]:
    """Every element the sweep must fit, read from the code that declares it.

    Returns
    -------
    frozenset of str
        Tagged names: ``method:<assessment family>/<fitted method>`` from the capability rows,
        ``engine:<engine>`` from the registered support rows, ``family:<id>`` from the target
        family maps, the theory anchors, the regimen kinds and the support rows, and
        ``estimand:<name>`` from the registered point targets.
    """
    rows = registered_rows()
    methods = {
        f"method:{row.result_family}/{method}"
        for row in ASSESSMENT_CAPABILITIES
        for method in row.methods
    }
    engines = {_engine(row.key.engine) for row in rows}
    families = {
        *FAMILY_THEORY,
        *_support._ARM_FAMILIES.values(),
        *_support._AXIS_FAMILIES.values(),
        _support.OPTIMAL_RULE_FAMILY,
        _support.LOG_CRR_PROJECTION_FAMILY,
        *REGIMEN_KINDS,
        *(family for row in rows for family in row.key.family),
    }
    estimands = {f"estimand:{name}" for name in TARGETS}
    return frozenset({*methods, *engines, *(f"family:{family}" for family in families), *estimands})


def covered(result: Any) -> frozenset[str]:
    """The tagged elements one fitted result stands for, read from the result itself.

    Parameters
    ----------
    result : object
        A fitted result of a kind.

    Returns
    -------
    frozenset of str
        Its method, and, where its configuration carries them, its support engine, target
        families and requested point estimands.
    """
    found = {f"method:{result.assessment_family}/{result.fitted_method}"}
    config = getattr(result, "config", None)
    key = getattr(config, "support_key", None)
    if key is not None:
        found.add(_engine(key.engine))
        found.update(f"family:{family}" for family in key.family)
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
    found = [f"{name}: {_KINDS_MISSING}" for name in sorted(elements - fitted - set(exempt))]
    found += [
        f"{name}: exempt, and no longer in the code" for name in sorted(set(exempt) - elements)
    ]
    found += [f"{name}: exempt, and a kind now fits it" for name in sorted(set(exempt) & fitted)]
    return found


def test_every_estimator_is_a_kind_of_the_sweep_or_an_owned_exemption() -> None:
    assert gaps(universe(), _swept_kinds(), UNSWEPT) == []


def test_every_exemption_names_a_reason_and_an_owner() -> None:
    for name, exemption in UNSWEPT.items():
        assert exemption.reason.strip() and exemption.owner.strip(), name


def test_the_universe_reads_every_axis() -> None:
    """A universe that lost an axis would cover nothing on it and pass."""
    elements = universe()
    for tag in ("method:", "engine:", "family:", "estimand:"):
        assert any(name.startswith(tag) for name in elements), tag
    assert "method:point/tmle" in elements
    assert "family:P1" in elements and "estimand:ate" in elements


#: Each control changes one input and must make :func:`gaps` report the named element.
_CONTROLS: dict[str, tuple[Callable[[frozenset[str]], frozenset[str]], dict[str, Exemption], str]]
_CONTROLS = {
    "a kind is removed": (
        lambda fitted: fitted - {"method:point/adaptive_iptw", "engine:adaptive_iptw"},
        dict(UNSWEPT),
        "method:point/adaptive_iptw",
    ),
    "a new engine appears with no kind": (
        lambda fitted: fitted,
        dict(UNSWEPT),
        "engine:new_engine",
    ),
    "an exemption outlives its gap": (
        lambda fitted: fitted,
        {**UNSWEPT, "method:point/tmle": Exemption(_KINDS_MISSING, _FIX15)},
        "method:point/tmle",
    ),
    "an exemption names nothing the code has": (
        lambda fitted: fitted,
        {**UNSWEPT, "family:P99": Exemption(_KINDS_MISSING, _FIX15)},
        "family:P99",
    ),
}


@pytest.mark.parametrize("name", list(_CONTROLS))
def test_each_control_makes_the_check_fail(name: str) -> None:
    change, exempt, element = _CONTROLS[name]
    elements = universe() | ({"engine:new_engine"} if element == "engine:new_engine" else set())
    found = gaps(elements, change(_swept_kinds()), exempt)
    assert any(line.startswith(f"{element}:") for line in found), found
