"""``dataclasses.replace`` works on every design, whatever the declared fields hold.

A frozen design stores some declared fields in a normalized form: a mapping becomes a tuple of
pairs, so that the design is hashable.  ``replace`` hands every stored field back to
``__init__``, so each normalizer has to read the form it stores.  Before this check
``replace`` on a declared ``LongPanel`` raised ``AttributeError`` for any field, and on
``PointTreatment``, ``TimeToEvent`` and ``LongitudinalTreatment`` it raised ``DataError``.

Each case below replaces one field and compares the result with the design built directly from
the same arguments.  An equality that held because both sides ignored the replaced value would
pass, so the fit checks replace the declared mechanism and show that the fit moves.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from cleverly import (
    CausalStudy,
    IPWMethod,
    LongitudinalTreatment,
    LongPanel,
    ModelSpec,
    PointTreatment,
    RegimeContrast,
    Runtime,
    TimeToEvent,
)
from cleverly.datasets import OFFER_WEEKS, make_sequential_offers
from cleverly.exceptions import DataError
from cleverly.learners import GLM

LOGGED = {"none": "p_none", "discount": "p_discount", "shipping": "p_shipping"}
#: A permutation of the logged columns.  Each row still sums to one, and the fit moves.
SWAPPED = {"none": "p_discount", "discount": "p_shipping", "shipping": "p_none"}


def _point() -> PointTreatment:
    return PointTreatment(
        outcome="y",
        treatment="a",
        adjustment=["x"],
        strata=["s"],
        treatment_probabilities={0: "p0", 1: "p1"},
    )


def _wide() -> LongitudinalTreatment:
    return LongitudinalTreatment(
        outcome="y",
        treatment=["a1", "a2"],
        baseline=["x"],
        time_varying=[[], ["m"]],
        censoring=["c1", "c2"],
        treatment_probabilities={"a1": {0: "p10", 1: "p11"}, "a2": {0: "p20", 1: "p21"}},
        censoring_probabilities={"c1": "q1", "c2": "q2"},
    )


def _time_to_event() -> TimeToEvent:
    return TimeToEvent(
        time="t",
        event="e",
        treatment="a",
        baseline=["x"],
        grid=[1, 2],
        causes={1: "relapse", 2: "death"},
        treatment_probabilities={0: "p0", 1: "p1"},
    )


def _panel(**overrides: Any) -> LongPanel:
    fields: dict[str, Any] = {
        "unit": "customer",
        "time": "week",
        "treatment": "offer",
        "baseline": ["tenure"],
        "time_varying": ["engaged"],
        "censoring": "consent",
        "outcome": "full_price",
        "times": [1, 2, 3],
        "treatment_probabilities": LOGGED,
        "censoring_probabilities": "p_consent",
    }
    return LongPanel(**{**fields, **overrides})


def _event_panel() -> LongPanel:
    return _panel(outcome=None, event={"relapse": "r", "death": "d"})


#: Each case: a name, the design, and the replacements to try, as keyword arguments.
CASES: list[tuple[str, Any, list[dict[str, Any]]]] = [
    (
        "point",
        _point,
        [
            {"adjustment": ["x", "z"]},
            {"strata": []},
            {"treatment_probabilities": {0: "q0", 1: "q1"}},
            {"treatment_probabilities": None},
            {"weights": "w"},
        ],
    ),
    (
        "wide",
        _wide,
        [
            {"baseline": ["x", "z"]},
            {"time_varying": [["u"], ["m"]]},
            {"treatment": ["a1", "a2", "a3"], "treatment_probabilities": None},
            {"censoring": ["c1"], "censoring_probabilities": {"c1": "q1"}},
            {"treatment_probabilities": {"a1": {0: "r10", 1: "r11"}}},
            {"censoring_probabilities": {"c1": "r1", "c2": "r2"}},
            {"treatment_probabilities": None, "censoring_probabilities": None},
            {"continuous_treatment": ["a1"], "treatment_probabilities": None},
        ],
    ),
    (
        "time_to_event",
        _time_to_event,
        [
            {"grid": [1, 2, 3]},
            {"causes": {1: "other"}},
            {"baseline": ["x", "z"]},
            {"treatment_probabilities": {0: "q0", 1: "q1"}},
            {"treatment_probabilities": None},
        ],
    ),
    (
        "panel_outcome",
        _panel,
        [
            {"baseline": ["tenure", "age"]},
            {"time_varying": []},
            {"times": [1, 2]},
            {"treatment": ["offer", "dose"], "treatment_probabilities": None},
            {"treatment_probabilities": SWAPPED},
            {"treatment_probabilities": None},
            {"censoring_probabilities": None},
            {"weights": "w"},
        ],
    ),
    (
        "panel_event",
        _event_panel,
        [
            {"event": {"relapse": "r2", "death": "d2"}},
            {"event": {"relapse": "r"}},
            {"event": "any", "treatment_probabilities": None},
            {"treatment_probabilities": SWAPPED},
        ],
    ),
]


def _cases() -> list[Any]:
    return [
        pytest.param(build, change, id=f"{name}-{'+'.join(change)}")
        for name, build, changes in CASES
        for change in changes
    ]


@pytest.mark.parametrize("build", [build for _, build, _ in CASES])
def test_replace_with_no_change_returns_an_equal_design(build: Any) -> None:
    design = build()
    assert dataclasses.replace(design) == design


@pytest.mark.parametrize("build", [build for _, build, _ in CASES])
def test_every_field_survives_a_replace_by_its_own_stored_value(build: Any) -> None:
    """The stored form of a field is an accepted input, for each initialized field."""
    design = build()
    for field in dataclasses.fields(design):
        if field.init:
            assert (
                dataclasses.replace(design, **{field.name: getattr(design, field.name)}) == design
            )


@pytest.mark.parametrize(("build", "change"), _cases())
def test_replace_equals_the_design_built_directly(build: Any, change: dict[str, Any]) -> None:
    """A replaced design equals the one built from the same arguments, field by field."""
    design = build()
    replaced = dataclasses.replace(design, **change)
    direct = type(design)(
        **{
            **{f.name: getattr(design, f.name) for f in dataclasses.fields(design) if f.init},
            **change,
        }
    )
    assert replaced == direct
    # The replaced value reached the design: each replaced field differs from the original.
    changed = [name for name, value in change.items() if value != getattr(design, name, None)]
    assert any(getattr(replaced, name) != getattr(design, name) for name in changed)


def test_a_replaced_mapping_is_read_back_as_the_mapping_it_was() -> None:
    """The stored pairs of the three mapping fields hold the replaced values, in order."""
    panel = dataclasses.replace(_panel(), treatment_probabilities=SWAPPED)
    assert dict(panel.treatment_probabilities) == SWAPPED
    wide = dataclasses.replace(_wide(), censoring_probabilities={"c2": "z2"})
    assert dict(wide.censoring_probabilities) == {"c2": "z2"}
    nested = dataclasses.replace(_wide(), treatment_probabilities={"a2": {1: "k"}})
    assert {node: dict(pairs) for node, pairs in nested.treatment_probabilities} == {"a2": {1: "k"}}


def test_a_value_that_is_not_a_declaration_is_still_refused() -> None:
    """Reading the stored form does not widen the input past a mapping of columns."""
    with pytest.raises(DataError, match="treatment_probabilities"):
        dataclasses.replace(_point(), treatment_probabilities=[0.5, 0.5])
    with pytest.raises(DataError, match="treatment_probabilities"):
        dataclasses.replace(_time_to_event(), treatment_probabilities=3)
    with pytest.raises(DataError, match="treatment_probabilities"):
        dataclasses.replace(_wide(), treatment_probabilities={"a1": [0.5]})


def _ipw_psi(panel: Any, design: LongPanel) -> dict[str, float]:
    plans = {name: (name,) * OFFER_WEEKS for name in ("none", "discount")}
    effect = CausalStudy(panel, design=design).identify(RegimeContrast(plans, reference="none"))
    result = effect.estimate(
        method=IPWMethod(
            models=ModelSpec(censoring_learner=GLM()), runtime=Runtime(random_state=1, n_jobs=1)
        )
    )
    return {key: estimate.psi for key, estimate in result.estimates.items()}


def test_a_replaced_panel_fits_identically_to_one_built_directly() -> None:
    """The fit reads the replaced mechanism, not the original, and not a copy of it.

    The permuted columns change the declared offer probabilities, so the fit with them differs
    from the fit with the logged ones.  A ``replace`` that kept the old mapping would fail the
    second assertion, and one that dropped it would fail the first.
    """
    panel, _ = make_sequential_offers(4000, seed=1)
    base = {
        "unit": "customer",
        "time": "week",
        "treatment": "offer",
        "baseline": ["tenure"],
        "time_varying": ["engaged"],
        "censoring": "consent",
        "outcome": "full_price",
    }
    original = LongPanel(**base, treatment_probabilities=LOGGED)
    replaced = dataclasses.replace(original, treatment_probabilities=SWAPPED)
    direct = LongPanel(**base, treatment_probabilities=SWAPPED)
    assert _ipw_psi(panel, replaced) == _ipw_psi(panel, direct)
    assert _ipw_psi(panel, replaced) != _ipw_psi(panel, original)
    undeclared = dataclasses.replace(original, treatment_probabilities=None)
    assert undeclared == LongPanel(**base)
