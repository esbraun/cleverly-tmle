"""A typed estimand holds one intervention kind, and ``CausalStudy.identify`` checks it.

``RegimeMean`` and ``RegimeContrast`` hold treatment levels and regimes,
``ModifiedTreatmentPolicy`` and ``ModifiedTreatmentPolicyEffect`` hold ``Shift`` objects, and
``IncrementalMean`` and ``IncrementalEffect`` hold ``Incremental`` objects.  Before RM14 in
``docs/roadmap.md``, no field checked its items.  A mixed request passed ``identify``, and
``estimate`` refused it, crashed with ``AttributeError`` after a learner fit, or read a
``Shift`` in ``IncrementalEffect`` as an odds multiplier and reported a number.  This module
pins these things:

* ``identify`` refuses an item of another kind with ``CapabilityError``, names the field, the
  item and the typed estimands for its kind, and cites F17;
* ``identify`` refuses a mapping, a string or a bare item as a point set with ``DataError``;
* no learner fits before either refusal;
* the estimator keeps a second guard on ``interventions=``, ``shifts=`` and ``incremental=``;
* a set of one kind still identifies, and each keyword of one kind still constructs;
* a mutation that removes the identification check fails every ``identify`` witness, and a
  mutation that removes every check lets a learner fit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

import cleverly.study as study_module
from cleverly import (
    CausalStudy,
    IncrementalEffect,
    IncrementalMean,
    ModifiedTreatmentPolicy,
    ModifiedTreatmentPolicyEffect,
    PointTreatment,
    RegimeContrast,
    RegimeMean,
)
from cleverly.datasets import make_shift_dose
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError, DataError
from cleverly.interventions import Incremental, Rule, Shift, Static, Stochastic
from cleverly.interventions import base as base_module
from tests import discrete_law as law
from tests.unit._declaration_support import assert_every_witness_fails, assert_refused, tmle_module
from tests.unit._natural_course_support import NeverFit, never_fit_learners

F17 = "docs/roadmap.md F17"
REGIME = "RegimeMean and RegimeContrast"
SHIFT = "ModifiedTreatmentPolicy and ModifiedTreatmentPolicyEffect"
INCREMENTAL = "IncrementalMean and IncrementalEffect"
#: The remedy for a bare value in a shift or incremental set.
BARE = "is a bare value"


def binary() -> CausalStudy:
    """The discrete law: binary ``A``, one covariate ``W``."""
    design = PointTreatment(outcome="Y", treatment="A", adjustment=("W",))
    return CausalStudy(law.frame(), design=design)


def dose() -> CausalStudy:
    """A continuous dose ``A`` with three covariates."""
    frame, _ = make_shift_dose(n=120, seed=3)
    design = PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=("W1", "W2", "W3"),
        treatment_kind="continuous",
    )
    return CausalStudy(frame, design=design)


@dataclass(frozen=True)
class Request:
    """One malformed point request, and what its refusal at ``identify`` must say.

    Parameters
    ----------
    study : callable
        Builds the study that identifies ``estimand``.
    estimand : Any
        The typed estimand.
    error : type
        The refusal: ``CapabilityError`` for an item of another kind, ``DataError`` for a
        set that is not a sequence.
    holder : str
        The field that the message names.
    route : tuple of str
        Fragments that every guard writes: the typed estimands and F17 for a mixed kind,
        and the sequence form for a malformed set.
    fits_before_rm14 : int
        The learner fits that commit 0cdab190 ran before its error.  A spy can fail only
        when it is positive.
    """

    study: Callable[[], CausalStudy]
    estimand: Any
    error: type[Exception]
    holder: str
    route: tuple[str, ...]
    fits_before_rm14: int


MIXED = {
    "incremental in a regimen set": Request(
        binary,
        RegimeContrast(regimens=(Static(1), Incremental(2.0))),
        CapabilityError,
        "RegimeContrast.regimens",
        (INCREMENTAL, "functional of P", F17),
        0,
    ),
    "shift in a regimen set": Request(
        binary,
        RegimeMean(regimens=(Static(1), Shift(0.5, cap=None))),
        CapabilityError,
        "RegimeMean.regimens",
        (SHIFT, "d(A, W)", "covariates alone", F17),
        0,
    ),
    "regime in an incremental set": Request(
        binary,
        IncrementalEffect((Incremental(1.0), Static(1))),
        CapabilityError,
        "IncrementalEffect.interventions",
        (REGIME, F17),
        1,
    ),
    "regime alone in an incremental set": Request(
        binary,
        IncrementalMean((Static(1),)),
        CapabilityError,
        "IncrementalMean.interventions",
        (REGIME, F17),
        1,
    ),
    "shift in an incremental set": Request(
        binary,
        IncrementalEffect((Incremental(1.0, name="one"), Shift(0.5, cap=None, name="s"))),
        CapabilityError,
        "IncrementalEffect.interventions",
        (SHIFT, F17),
        2,
    ),
    "level in an incremental set": Request(
        binary,
        IncrementalMean((Incremental(1.0), 2.0)),
        CapabilityError,
        "IncrementalMean.interventions",
        (BARE, "Write it as an object, such as Incremental(2.0)"),
        0,
    ),
    "incremental in a shift set": Request(
        dose,
        ModifiedTreatmentPolicy(shifts=(Shift(0.5, cap=None, name="s"), Incremental(2.0))),
        CapabilityError,
        "ModifiedTreatmentPolicy.shifts",
        (INCREMENTAL, F17),
        1,
    ),
    "regime in a shift set": Request(
        dose,
        ModifiedTreatmentPolicyEffect(shifts=(Shift(0.5, cap=None, name="s"), Static(1))),
        CapabilityError,
        "ModifiedTreatmentPolicyEffect.shifts",
        (REGIME, F17),
        1,
    ),
}

#: A set that is not a sequence.  The constructor guards read a tuple the estimator has
#: already built, so ``identify`` is the one guard, and ``route`` is its sequence form.
MALFORMED = {
    "bare regime": Request(
        binary,
        RegimeMean(regimens=Static(1)),
        DataError,
        "RegimeMean.regimens",
        ("must be a sequence", "one-item tuple"),
        0,
    ),
    "bare incremental": Request(
        binary,
        IncrementalMean(Incremental(2.0)),
        DataError,
        "IncrementalMean.interventions",
        ("must be a sequence", "one-item tuple"),
        0,
    ),
    "bare string level": Request(
        binary,
        RegimeMean(regimens="1"),
        DataError,
        "RegimeMean.regimens",
        ("must be a sequence", "one-item tuple"),
        0,
    ),
    "zero-dimensional array": Request(
        binary,
        RegimeMean(regimens=np.array(1)),
        DataError,
        "RegimeMean.regimens",
        ("must be a sequence", "one-item tuple"),
        0,
    ),
    "generator": Request(
        binary,
        RegimeMean(regimens=(level for level in (0, 1))),
        DataError,
        "RegimeMean.regimens",
        ("must be a sequence", "iterator", "empty after its first read"),
        1,
    ),
    "mapping": Request(
        binary,
        RegimeMean(regimens={"a": 1, "b": 0}),
        DataError,
        "RegimeMean.regimens",
        ("must be a sequence", "is a mapping", "name="),
        2,
    ),
}

REQUESTS = {**MIXED, **MALFORMED}

#: The rows whose error came after a learner fit at 0cdab190, so a spy can fail on them.
SPIED = [name for name, row in REQUESTS.items() if row.fits_before_rm14 > 0]


def identify(row: Request) -> Any:
    return row.study().identify(row.estimand)


def estimate_with_spies(row: Request) -> Any:
    """``identify`` then ``estimate`` in sample, with a ``NeverFit`` in every learner slot."""
    learners = never_fit_learners()
    return identify(row).estimate(
        outcome_learner=learners["outcome_learner"],
        treatment_learner=learners["treatment_learner"],
        cross_fit=False,
        simultaneous=False,
    )


def identify_witness(name: str) -> None:
    row = REQUESTS[name]
    assert_refused(lambda: identify(row), row.error, row.holder, *row.route)


def spy_witness(name: str) -> None:
    """The refusal precedes every learner.  It checks ``route`` and not ``holder``, so the
    constructor guard satisfies it when the identification check is gone."""
    row = REQUESTS[name]
    assert_refused(lambda: estimate_with_spies(row), row.error, *row.route)
    assert NeverFit.calls == 0, f"{NeverFit.calls} learner fit(s) ran before the refusal"


class TestAMixedRequestIsRefusedAtIdentify:
    @pytest.mark.parametrize("name", list(MIXED))
    def test_identify_refuses_the_item_by_its_typed_estimand(self, name: str) -> None:
        identify_witness(name)
        assert_refused(lambda: identify(MIXED[name]), CapabilityError, "item ")

    @pytest.mark.parametrize("name", list(MALFORMED))
    def test_identify_refuses_a_set_that_is_not_a_sequence(self, name: str) -> None:
        identify_witness(name)

    @pytest.mark.parametrize("name", [name for name in MALFORMED if name != "mapping"])
    def test_only_a_mapping_gets_the_mapping_remedy(self, name: str) -> None:
        with pytest.raises(DataError) as raised:
            identify(MALFORMED[name])
        assert "mapping" not in str(raised.value)

    def test_a_refused_generator_is_not_read(self) -> None:
        """At commit 3e67b783 the check read the generator, and the functional was empty."""
        plans = (level for level in (0, 1))
        assert_refused(lambda: binary().identify(RegimeMean(regimens=plans)), DataError)
        assert next(plans) == 0

    @pytest.mark.parametrize("name", SPIED)
    def test_no_learner_fits_before_the_refusal(self, name: str) -> None:
        spy_witness(name)

    def test_no_message_calls_an_intervention_something_else(self) -> None:
        for row in MIXED.values():
            with pytest.raises(CapabilityError) as raised:
                identify(row)
            assert "not an intervention" not in str(raised.value)
            assert "not as an intervention" not in str(raised.value)


class TestTheEstimatorKeepsASecondGuard:
    @pytest.mark.parametrize(
        ("keyword", "items", "fragments"),
        [
            (
                "interventions",
                (Static(0), Incremental(2.0)),
                ("interventions=", "item 2", INCREMENTAL, "TMLE(incremental=...)", F17),
            ),
            (
                "interventions",
                (Shift(0.5, cap=None),),
                ("interventions=", "item 1", SHIFT, "TMLE(shifts=...)", F17),
            ),
            (
                "incremental",
                (Incremental(1.0), Static(1)),
                ("incremental=", "item 2", REGIME, "TMLE(interventions=...)", F17),
            ),
            (
                "incremental",
                (Incremental(1.0), Shift(0.5, cap=None)),
                ("incremental=", "item 2", SHIFT, F17),
            ),
            (
                "shifts",
                (Shift(0.5, cap=None), Incremental(2.0)),
                ("shifts=", "item 2", INCREMENTAL, F17),
            ),
            (
                "shifts",
                (0.5,),
                ("shifts=", "item 1", BARE, "Shift(0.5, cap=None)"),
            ),
        ],
        ids=[
            "incremental in interventions",
            "shift in interventions",
            "regime in incremental",
            "shift in incremental",
            "incremental in shifts",
            "bare number in shifts",
        ],
    )
    def test_the_constructor_refuses_a_mixed_keyword(
        self, keyword: str, items: tuple[Any, ...], fragments: tuple[str, ...]
    ) -> None:
        assert_refused(lambda: TMLE(**{keyword: items}), CapabilityError, *fragments)


class TestAOneKindRequestStillIdentifies:
    def test_a_regimen_set_of_levels_and_regimes_identifies(self) -> None:
        regimens = (
            0,
            Static(1),
            Rule(lambda covariates: np.ones(len(covariates)), "treat all", rule_kind="known"),
            Stochastic(
                lambda covariates: np.full((len(covariates), 2), 0.5),
                "coin",
                density_kind="known",
            ),
        )
        effect = binary().identify(RegimeContrast(regimens=regimens))
        assert effect.functional.interventions == regimens

    def test_an_incremental_set_identifies(self) -> None:
        effect = binary().identify(IncrementalEffect((Incremental(0.5), Incremental(2.0))))
        assert effect.functional.axis == "ipsi"

    def test_a_shift_set_identifies_on_a_dose(self) -> None:
        shifts = (Shift(0.0, cap=None), Shift(0.5, cap=None))
        effect = dose().identify(ModifiedTreatmentPolicyEffect(shifts))
        assert effect.functional.interventions == shifts

    def test_each_keyword_of_one_kind_constructs(self) -> None:
        assert len(TMLE(interventions=(1, 0)).interventions) == 2
        assert len(TMLE(incremental=(Incremental(1.0), Incremental(2.0))).incremental) == 2
        assert len(TMLE(shifts=(Shift(0.5, cap=None),)).shifts) == 1


class TestTheWitnessesHaveTeeth:
    @staticmethod
    def remove(monkeypatch: pytest.MonkeyPatch, *modules: Any) -> None:
        for module in modules:
            monkeypatch.setattr(module, "refuse_mixed_interventions", lambda *a, **k: None)

    def test_removing_the_identify_check_fails_every_identify_witness(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The refusal then moves to ``estimate``, where the constructor guard still runs
        before every learner, so each mixed spy witness still passes."""
        self.remove(monkeypatch, study_module)
        assert_every_witness_fails(lambda name=name: identify_witness(name) for name in MIXED)
        for name in SPIED:
            if name in MIXED:
                spy_witness(name)

    def test_removing_every_check_lets_a_learner_fit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``NeverFit.fit`` raises ``AssertionError`` at the first learner, as 0cdab190 did."""
        self.remove(monkeypatch, study_module, tmle_module, base_module)
        spied = [name for name in SPIED if name in MIXED]
        assert spied
        assert_every_witness_fails(lambda name=name: spy_witness(name) for name in spied)
