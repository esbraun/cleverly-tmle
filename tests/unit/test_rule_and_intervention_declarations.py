r"""A treatment rule and a user-written intervention must be declared known.

A ``Rule`` assigns :math:`d(W)`, and a user-written ``Intervention`` returns its own
:math:`g^\star(a \mid W)` from the data of the fit.  When the function is fixed before the
fit, the regime influence curve that the package reports is the efficient influence function
of the regime mean.  When it is learned from the analysis sample, the target changes.  A
population-indexed rule, such as a threshold at the mean of :math:`W`, has a curve with a
further pathwise-derivative term that the regime curve lacks, and a realized learned rule
defines a data-adaptive target.

A callable can close over any estimate, and no code can inspect a closure, so the status of
the function is a declaration: ``Rule(rule_kind=...)``, or a ``density_kind`` attribute on a
user-written class.  RM28 in ``docs/roadmap.md`` records the defect.  This module pins these
things:

* the rule declaration is required, ``"estimated"`` is refused, and a rule that is not
  callable is refused, with their messages;
* the positional order of ``Rule`` is unchanged;
* the protocol carries the declaration: ``Static`` reads known by its exact type, and
  ``as_interventions`` admits a user-written class as it is, so its fit can refuse it;
* every fit entry refuses an undeclared class, and a restored or modified rule, before any
  learner or regime-function call, and so do ``RegimeSet.evaluate`` and ``Rule.density``;
* the declaration refusal comes before a refusal of the fit configuration;
* a rule pickled before the field existed loads undeclared, and a result restored with such a
  rule refuses every recomputation;
* the simulated-confounding replay refuses a restored rule before it runs, and each frozen
  regime carries the declaration of its source;
* a deliberate mutation that removes a check makes those witnesses fail;
* on exact laws, a threshold at the sample mean and a user-written class that tilts the
  sample mechanism, each declared ``"known"``, get the fixed-function curve, which
  understates the standard error of the population-indexed target;
* a fixed threshold and a user-written class with a known density keep their intervals.

``tests/unit/_policy_declaration_support.py`` holds the rules, the classes, and the
threshold law.  ``tests/unit/test_stochastic_regime_densities.py`` lists both declarations
among the users of the shared check.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields, replace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import cleverly.interventions.base as base_module
from cleverly import RegimeMean
from cleverly.data import CausalData
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError, DataError
from cleverly.interventions import (
    Intervention,
    RegimeSet,
    Rule,
    Static,
    Stochastic,
    as_interventions,
)
from cleverly.interventions.base import (
    _ESTIMATED_INTERVENTION,
    _ESTIMATED_RULE,
    _UNDECLARED_INTERVENTION,
    _UNDECLARED_RULE,
    refuse_regime_densities,
)
from cleverly.sensitivity import _simulated_confounding_fixed as replay_module
from cleverly.sensitivity import simulated_confounding
from tests import discrete_law as law
from tests.conftest import linear_in_sample
from tests.pickles import legacy_without
from tests.unit._confounding_support import Counter, forbid_draw_and_refit, validate_replay
from tests.unit._declaration_support import (
    PATHWISE,
    assert_every_witness_fails,
    assert_refused,
    assert_refused_before_any_call,
    oracle_fit,
    point_entries,
    recomputations,
    restored,
    restored_states,
    se_ratio,
    tmle_module,
)
from tests.unit._declaration_support import legacy_result as legacy_result_of
from tests.unit._natural_course_support import NeverFit, never_fit_learners
from tests.unit._policy_declaration_support import (
    CENTRE,
    EXACT_RATIO,
    INTERVENTION_UNKNOWN,
    PSI,
    RATIO_PIN,
    RULE_UNKNOWN,
    UNDERSTATEMENT_BOUND,
    BareTilt,
    DataTilt,
    FixedThreshold,
    KnownUserTilt,
    SampleThreshold,
    fixed_rule_curve,
    regime_value,
    sample_ratio,
    threshold_curve,
    threshold_fit,
    threshold_frame,
    threshold_rule,
    threshold_term,
)
from tests.unit._simulated_confounding_support import _GRID, _alias, _estimate, _fit_policy, _study
from tests.unit._tilt_law_support import (
    CELL_P,
    TILT_COUNTS,
    TILT_PROBS,
    KnownTilt,
    SampleTilt,
    fixed_eif,
    tilt_curve,
)
from tests.unit._tilt_law_support import UNDERSTATEMENT_BOUND as TILT_BOUND

#: The four declaration refusals, imported from the module that raises them, so each text
#: is written once.
UNDECLARED_RULE = _UNDECLARED_RULE
ESTIMATED_RULE = _ESTIMATED_RULE
UNDECLARED_CLASS = _UNDECLARED_INTERVENTION
ESTIMATED_CLASS = _ESTIMATED_INTERVENTION
#: Fragments of the other refusals.  A test matches a fragment, not the whole text, so a
#: rewording of the explanation does not break it, but a message from another check does.
NOT_CALLABLE = "Rule rule= must be callable"
NO_DENSITY = "an Intervention needs a density(data) method"
BARE_CALLABLE = "interventions= received a callable"
NO_NAME = "has a density method but no name"

#: The replay regime class, read before any test can replace the module global.
FROZEN = replay_module._FrozenRegime


def causal_data() -> CausalData:
    return CausalData.from_frame(law.frame(), outcome="Y", treatment="A")


# ------------------------------------------------------------------ the rule declaration


class TestTheRuleDeclarationIsRequired:
    def test_an_undeclared_rule_is_refused(self) -> None:
        assert_refused(threshold_rule, CapabilityError, UNDECLARED_RULE, "rule_kind='known'")

    def test_an_estimated_rule_is_refused_by_its_missing_term(self) -> None:
        assert_refused(
            lambda: threshold_rule(rule_kind="estimated"),
            CapabilityError,
            ESTIMATED_RULE,
            PATHWISE,
            "data-adaptive target",
            "nonregular at ties",
        )

    @pytest.mark.parametrize("kind", ["Known", "probability", True, 1])
    def test_an_unknown_declaration_is_refused(self, kind: Any) -> None:
        assert_refused(lambda: threshold_rule(rule_kind=kind), DataError, RULE_UNKNOWN)

    @pytest.mark.parametrize("rule", [0.6, np.ones(10)], ids=["float", "array"])
    @pytest.mark.parametrize("kind", [None, "known", "estimated"])
    def test_a_rule_that_is_not_callable_is_refused(self, rule: Any, kind: Any) -> None:
        """The callable check runs first, so every declaration meets the same refusal."""
        assert_refused(lambda: threshold_rule(rule, rule_kind=kind), DataError, NOT_CALLABLE)

    def test_a_known_rule_is_accepted_and_evaluates(self) -> None:
        rule = threshold_rule(rule_kind="known")
        assert rule.rule_kind == "known"
        data = causal_data()
        treated = (data.covariates[:, 0] <= CENTRE).astype(float)
        np.testing.assert_array_equal(rule.density(data)[:, 1], treated)

    def test_a_subclass_inherits_the_declaration(self) -> None:
        class Declared(Rule):
            pass

        assert_refused(lambda: Declared(FixedThreshold(), "thr"), CapabilityError, UNDECLARED_RULE)
        assert Declared(FixedThreshold(), "thr", rule_kind="known").rule_kind == "known"


class TestThePositionalOrderIsUnchanged:
    """``rule_kind`` is the last field, so ``Rule(rule, name)`` keeps its order."""

    def test_the_declaration_is_the_last_field(self) -> None:
        assert [field.name for field in fields(Rule)] == ["rule", "name", "rule_kind"]

    def test_the_third_positional_argument_is_the_declaration(self) -> None:
        assert Rule(FixedThreshold(), "thr", "known").rule_kind == "known"
        assert_refused(lambda: Rule(FixedThreshold(), "thr"), CapabilityError, UNDECLARED_RULE)


# ------------------------------------------------------------------ the protocol


class StaticSubclass(Static):
    """A subclass can override ``density``, so it reads as undeclared."""


class DeclaredStaticSubclass(Static):
    """A subclass that declares its own density."""

    density_kind = "known"


class TestTheProtocolCarriesTheDeclaration:
    def test_static_is_known_by_its_exact_type(self) -> None:
        assert Static(1).density_kind == "known"
        assert refuse_regime_densities((Static(1),)) is None

    def test_a_static_subclass_is_undeclared(self) -> None:
        assert StaticSubclass(1).density_kind is None
        assert_refused(
            lambda: refuse_regime_densities((StaticSubclass(1),)), CapabilityError, UNDECLARED_CLASS
        )
        assert DeclaredStaticSubclass(1).density_kind == "known"
        assert refuse_regime_densities((DeclaredStaticSubclass(1),)) is None

    def test_a_rule_reads_its_field(self) -> None:
        rule = threshold_rule(rule_kind="known")
        assert rule.density_kind == "known"
        assert restored(rule, "rule_kind", "estimated").density_kind == "estimated"

    def test_every_declared_regime_satisfies_the_protocol(self) -> None:
        """The runtime check needs ``density_kind``, so a class without it fails the check."""
        declared = [
            Static(1),
            threshold_rule(rule_kind="known"),
            Stochastic(lambda w: np.ones((len(w), 2)) / 2, "coin", density_kind="known"),
            FROZEN("thr", np.ones((2, 2)) / 2, None, density_kind="known"),
            DataTilt(density_kind="known"),
        ]
        for item in declared:
            assert isinstance(item, Intervention), type(item).__name__
        assert not isinstance(BareTilt(), Intervention)


class TestAUserWrittenInterventionIsAdmitted:
    @pytest.mark.parametrize("item", [DataTilt(), BareTilt()], ids=["declared field", "bare"])
    def test_a_class_with_a_density_is_taken_as_is(self, item: Any) -> None:
        """Not wrapped in ``Static``, whose level would be the object itself."""
        (admitted,) = as_interventions((item,))
        assert admitted is item

    def test_a_bare_callable_is_refused_with_the_rule_remedy(self) -> None:
        assert_refused(
            lambda: as_interventions((lambda w: np.ones(len(w)),)),
            DataError,
            BARE_CALLABLE,
            "<lambda>",
            "Rule(rule, name, rule_kind='known')",
        )
        assert_refused(
            lambda: as_interventions((FixedThreshold(),)),
            DataError,
            BARE_CALLABLE,
            "FixedThreshold",
        )

    def test_a_class_with_a_density_and_no_name_is_refused(self) -> None:
        class Nameless:
            density_kind = "known"

            def density(self, data: Any) -> np.ndarray:
                raise AssertionError("never evaluated")

        assert_refused(lambda: as_interventions((Nameless(),)), DataError, "Nameless", NO_NAME)

    def test_levels_still_read_as_static(self) -> None:
        """The control: a level, a string, and an object with no density stay ``Static``."""
        items = as_interventions((1, "high", 0.5))
        assert [type(item) for item in items] == [Static, Static, Static]
        assert [item.level for item in items] == [1, "high", 0.5]  # type: ignore[attr-defined]

    def test_an_object_without_a_density_method_is_refused_by_the_check(self) -> None:
        """A restored estimator can hold anything, so the check reads the method first."""
        assert_refused(lambda: refuse_regime_densities((object(),)), DataError, NO_DENSITY)


# ------------------------------------------------------------------ the fit layer


def spy_rule() -> Rule:
    """A known rule whose function counts its calls in ``rule.rule.calls``."""
    return threshold_rule(Counter(FixedThreshold()), rule_kind="known")


#: Every fit entry that can reach a declared object: ``TMLE.fit``, ``CausalStudy.estimate``,
#: and ``TMLE.refit``.
ENTRIES = point_entries(lambda item: {"interventions": (item,)}, lambda item: RegimeMean((item,)))

#: What a restored object can carry, and the refusal each one meets.
RESTORED_RULE = restored_states(UNDECLARED_RULE, ESTIMATED_RULE)
RESTORED_CLASS = restored_states(UNDECLARED_CLASS, ESTIMATED_CLASS)


def spy_of(item: Any) -> Any:
    """The object that counts the regime-function calls: the rule, or the class itself."""
    return item.rule if isinstance(item, Rule) else item


def assert_entry_refuses(entry: str, item: Any, *fragments: str) -> None:
    assert_refused_before_any_call(
        lambda: ENTRIES[entry](item, never_fit_learners()), spy_of(item), "regime", *fragments
    )


def undeclared_classes() -> dict[str, tuple[Any, tuple[str, ...]]]:
    """Each undeclared or estimated user class, and the fragments of the refusal it meets."""
    return {
        "no attribute": (BareTilt(), (UNDECLARED_CLASS, "density_kind attribute of 'known'")),
        **{
            name: (DataTilt(density_kind=kind), fragments)
            for name, (kind, fragments) in RESTORED_CLASS.items()
        },
    }


class TestTheFitRefusesAnUndeclaredIntervention:
    @pytest.mark.parametrize("entry", list(ENTRIES))
    @pytest.mark.parametrize("name", ["no attribute", *RESTORED_CLASS])
    def test_every_entry_refuses_before_any_learner_or_density_call(
        self, entry: str, name: str
    ) -> None:
        item, fragments = undeclared_classes()[name]
        assert_entry_refuses(entry, item, *fragments)

    def test_the_estimated_refusal_names_the_incremental_axis(self) -> None:
        _, fragments = RESTORED_CLASS["estimated"]
        assert_entry_refuses(
            "fit", DataTilt(density_kind="estimated"), *fragments, "TMLE(incremental=...)"
        )

    def test_an_unknown_declaration_is_refused_before_any_call(self) -> None:
        item = DataTilt(density_kind="Known")
        assert_refused_before_any_call(
            lambda: ENTRIES["fit"](item, never_fit_learners()),
            item,
            "density",
            INTERVENTION_UNKNOWN,
            error=DataError,
        )

    @pytest.mark.parametrize("entry", list(ENTRIES))
    def test_a_known_declaration_reaches_the_first_learner(self, entry: str) -> None:
        """The control: the same entries with the declaration intact go past the check."""
        item = DataTilt(density_kind="known")
        with pytest.raises(AssertionError, match="before any learner is fitted"):
            ENTRIES[entry](item, never_fit_learners())
        assert NeverFit.calls == 1
        assert item.calls == 0, "a density ran before the first learner"


class TestTheFitRefusesARestoredRule:
    @pytest.mark.parametrize("entry", list(ENTRIES))
    @pytest.mark.parametrize("name", list(RESTORED_RULE))
    def test_every_entry_refuses_before_any_learner_or_rule_call(
        self, entry: str, name: str
    ) -> None:
        kind, fragments = RESTORED_RULE[name]
        assert_entry_refuses(entry, restored(spy_rule(), "rule_kind", kind), *fragments)

    def test_a_malformed_rule_is_refused_before_any_call(self) -> None:
        """A rule that is not callable, and an unknown declaration, are data errors."""
        for field, value, fragment in [
            ("rule", np.ones(3), NOT_CALLABLE),
            ("rule_kind", "Known", RULE_UNKNOWN),
        ]:
            rule = spy_rule()
            spy = rule.rule
            restored(rule, field, value)
            assert_refused_before_any_call(
                lambda rule=rule: ENTRIES["fit"](rule, never_fit_learners()),
                spy,
                "rule",
                fragment,
                error=DataError,
            )

    def test_a_subclass_that_skips_the_declaration_refuses_at_the_fit(self) -> None:
        """The fit selects rules with ``isinstance``, so a subclass cannot opt out."""

        class Skips(Rule):
            def __post_init__(self) -> None:
                pass

        rule = Skips(Counter(FixedThreshold()), "thr")
        assert rule.rule_kind is None
        assert_entry_refuses("fit", rule, UNDECLARED_RULE)

    @pytest.mark.parametrize("entry", list(ENTRIES))
    def test_a_known_declaration_reaches_the_first_learner(self, entry: str) -> None:
        """The control: the same entries with the declaration intact go past the check."""
        rule = spy_rule()
        with pytest.raises(AssertionError, match="before any learner is fitted"):
            ENTRIES[entry](rule, never_fit_learners())
        assert NeverFit.calls == 1
        assert rule.rule.calls == 0, "a rule ran before the first learner"


def cross_fitted(item: Any, frame: pd.DataFrame, delta: str | None = None) -> Any:
    """A cross-fitted ``TMLE.fit`` of ``item`` on ``frame``, with ``NeverFit`` learners."""
    estimator = TMLE(
        interventions=(item,), cross_fit=True, n_folds=2, simultaneous=False, **never_fit_learners()
    )
    return estimator.fit(frame, outcome="Y", treatment="A", delta=delta)


def continuous_outcome() -> pd.DataFrame:
    """The default law with the outcome ``Y + W / 2``, which reads as continuous."""
    frame = law.frame()
    return frame.assign(Y=frame["Y"] + 0.5 * frame["W"])


def missing_outcome() -> pd.DataFrame:
    """The default law with every seventh outcome missing, and ``D`` its indicator."""
    frame = law.frame()
    observed = np.arange(len(frame)) % 7 != 0
    return frame.assign(D=observed.astype(float), Y=frame["Y"].where(observed))


#: Two fits that ``_resolve_estimands_for_data`` refuses after its regime check, as the fit,
#: the class of that refusal, and a fragment of its message.  ``cv_evaluation=True`` is not
#: among them: a regime fit with it reaches the first learner.
LATER_REFUSALS: dict[str, tuple[Callable[[Any], Any], type[Exception], str]] = {
    "unbounded scale": (
        lambda item: cross_fitted(item, continuous_outcome()),
        CapabilityError,
        "needs a declared q_bounds",
    ),
    "missing outcomes": (
        lambda item: cross_fitted(item, missing_outcome(), delta="D"),
        CapabilityError,
        "Cross-fitted TMLE with missing outcomes (delta=)",
    ),
}

#: Each declared object, and the same object restored undeclared with its refusal.
DECLARED_AND_RESTORED: dict[str, tuple[Callable[[], Any], Callable[[], Any], str]] = {
    "rule": (spy_rule, lambda: restored(spy_rule(), "rule_kind", None), UNDECLARED_RULE),
    "class": (lambda: DataTilt(density_kind="known"), DataTilt, UNDECLARED_CLASS),
}


class TestTheDeclarationRefusalComesFirst:
    """An undeclared object meets its own refusal, whatever else its fit configuration breaks.

    ``_resolve_estimands_for_data`` checks the regimes before any refusal of the fit
    configuration.  Each of those refusals names a remedy, and no remedy lets an undeclared
    function fit.  A fit that no such refusal stops reaches ``RegimeSet.evaluate`` only
    after the learners, so these tests are the witness of the fit-layer check.
    """

    @pytest.mark.parametrize("refusal", list(LATER_REFUSALS))
    @pytest.mark.parametrize("kind", list(DECLARED_AND_RESTORED))
    def test_a_declared_object_meets_the_later_refusal(self, refusal: str, kind: str) -> None:
        """The control: each configuration is refused when the declaration is intact."""
        fit, error, fragment = LATER_REFUSALS[refusal]
        item = DECLARED_AND_RESTORED[kind][0]()
        assert_refused_before_any_call(
            lambda: fit(item), spy_of(item), "regime", fragment, error=error
        )

    @pytest.mark.parametrize("refusal", list(LATER_REFUSALS))
    @pytest.mark.parametrize("kind", list(DECLARED_AND_RESTORED))
    def test_an_undeclared_object_meets_the_declaration_refusal(
        self, refusal: str, kind: str
    ) -> None:
        fit, _, _ = LATER_REFUSALS[refusal]
        _, build, undeclared = DECLARED_AND_RESTORED[kind]
        item = build()
        assert_refused_before_any_call(lambda: fit(item), spy_of(item), "regime", undeclared)


#: Each public evaluator that runs a rule, called directly as a user can call it.
EVALUATORS: dict[str, Callable[[Any], Any]] = {
    "RegimeSet.evaluate": lambda item: RegimeSet.evaluate((item,), causal_data()),
    "Rule.density": lambda item: item.density(causal_data()),
}


class TestTheEvaluatorsCheckFirst:
    """An undeclared object handed straight to an evaluator refuses before its function runs."""

    @pytest.mark.parametrize("evaluator", list(EVALUATORS))
    @pytest.mark.parametrize("name", list(RESTORED_RULE))
    def test_a_restored_rule_refuses_before_it_runs(self, evaluator: str, name: str) -> None:
        kind, fragments = RESTORED_RULE[name]
        rule = restored(spy_rule(), "rule_kind", kind)
        assert_refused(lambda: EVALUATORS[evaluator](rule), CapabilityError, *fragments)
        assert rule.rule.calls == 0, "the rule was evaluated before the refusal"

    @pytest.mark.parametrize("name", ["no attribute", *RESTORED_CLASS])
    def test_an_undeclared_class_refuses_before_its_density_runs(self, name: str) -> None:
        item, fragments = undeclared_classes()[name]
        evaluate = EVALUATORS["RegimeSet.evaluate"]
        assert_refused(lambda: evaluate(item), CapabilityError, *fragments)
        assert item.calls == 0, "the density was evaluated before the refusal"

    def test_the_set_checks_every_regime_before_the_first_function(self) -> None:
        """A declared rule listed first does not run before an undeclared class refuses."""
        first = spy_rule()
        item = DataTilt()
        assert_refused(
            lambda: RegimeSet.evaluate((first, item), causal_data()),
            CapabilityError,
            UNDECLARED_CLASS,
        )
        assert first.rule.calls == 0, "the first rule ran before the refusal"
        assert item.calls == 0

    @pytest.mark.parametrize("evaluator", list(EVALUATORS))
    def test_a_declared_rule_is_evaluated(self, evaluator: str) -> None:
        """The control: the same call with the declaration intact runs the rule."""
        rule = spy_rule()
        EVALUATORS[evaluator](rule)
        assert rule.rule.calls == 1

    def test_a_declared_class_is_evaluated(self) -> None:
        item = DataTilt(density_kind="known")
        EVALUATORS["RegimeSet.evaluate"](item)
        assert item.calls == 1


# ------------------------------------------------------------------ old pickles


def legacy(rule: Rule) -> Rule:
    """``rule`` as a pickle written before ``rule_kind`` existed would restore it."""
    return legacy_without(rule, "rule_kind")


class TestALegacyRuleLoads:
    def test_a_pickle_without_the_field_reads_none_and_can_be_replaced(self) -> None:
        old = legacy(threshold_rule(rule_kind="known"))
        assert "rule_kind" not in vars(old)
        assert old.rule_kind is None
        assert old.density_kind is None
        assert_refused(lambda: replace(old), CapabilityError, UNDECLARED_RULE)
        assert replace(old, rule_kind="known").rule_kind == "known"

    def test_a_legacy_rule_refuses_at_the_fit(self) -> None:
        """The pickle copies the counter, so the spy is the one on the restored rule."""
        assert_entry_refuses("fit", legacy(spy_rule()), UNDECLARED_RULE)

    def test_replacing_the_function_keeps_the_declaration(self) -> None:
        """The recorded limit: ``replace`` copies every field that the call does not name."""
        rule = replace(threshold_rule(rule_kind="known"), rule=SampleThreshold(0.4))
        assert rule.rule_kind == "known"


def rules(result: Any) -> list[Any]:
    return [item for item in result.estimator.interventions if isinstance(item, Rule)]


def legacy_result(result: Any) -> Any:
    """``result`` as an artifact written before ``rule_kind`` existed would restore it."""
    return legacy_result_of(result, "rule_kind", rules)


#: The estimands a retarget of the legacy result requests.
RETARGETED = ("ey_regime", "ate_regime")
RECOMPUTATIONS = ["truncation_curve", "retarget", "refit"]


@pytest.fixture(scope="module")
def rule_result() -> Any:
    """A declared ``Rule`` fit on the default law, beside a ``Static`` reference."""
    regimes = (Static(0, name="never"), threshold_rule(rule_kind="known"))
    estimator = TMLE(interventions=regimes, **linear_in_sample())
    return estimator.fit(law.frame(), outcome="Y", treatment="A").single()


class TestALegacyRuleResultRefusesARecomputation:
    """RM28: every recomputation from a restored result checks the rule as the fit does.

    Every sweep recomputes through ``_retarget_detailed``, and a refit through
    ``_resolve_estimands_for_data``.
    """

    @pytest.mark.parametrize("entry", RECOMPUTATIONS)
    def test_every_recomputation_refuses(self, rule_result: Any, entry: str) -> None:
        old = legacy_result(rule_result)
        assert_refused(recomputations(old, RETARGETED)[entry], CapabilityError, UNDECLARED_RULE)

    @pytest.mark.parametrize("entry", RECOMPUTATIONS)
    def test_the_declared_result_recomputes(self, rule_result: Any, entry: str) -> None:
        """The control: the same entry on the result before the declaration was lost."""
        assert "ey_regime[thr]" in rule_result.estimates
        recomputations(rule_result, RETARGETED)[entry]()


# ------------------------------------------------------------------ the replay


def counted_rule_fit() -> tuple[Any, Counter]:
    """A fitted regime mean whose rule counts its calls, restored without its declaration.

    The counter is read off the restored result, because a pickle round trip copies it.
    """
    rule = Rule(Counter(FixedThreshold(0.0)), name="policy", rule_kind="known")
    old = legacy_result(_estimate(_study(), RegimeMean((rule,))))
    function = old.estimator.interventions[0].rule
    function.calls = 0
    return old, function


def drop_the_carry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation R6: the replay builds each frozen regime with ``density_kind=None``."""

    def dropping(*arguments: Any, density_kind: Any = None) -> Any:
        return FROZEN(*arguments)

    monkeypatch.setattr(replay_module, "_FrozenRegime", dropping)


class TestTheReplay:
    def test_a_known_rule_replays_with_the_declaration_of_each_source(self) -> None:
        result = _fit_policy("rule")
        replay = validate_replay(result)
        sources = result.estimator.interventions
        assert [type(item) for item in sources] == [Static, Rule]
        assert all(type(item) is FROZEN for item in replay.interventions)
        assert [item.density_kind for item in replay.interventions] == [
            item.density_kind for item in sources
        ]
        assert [item.density_kind for item in replay.interventions] == ["known", "known"]
        assert refuse_regime_densities(replay.interventions) is None
        surface = simulated_confounding(
            result, estimand=_alias(result), grid=_GRID, random_state=31
        )
        assert all(cell.failure is None for cell in surface.cells)

    def test_a_legacy_rule_refuses_at_replay_before_it_runs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        old, function = counted_rule_fit()
        forbid_draw_and_refit(monkeypatch, old.estimator)
        assert_refused(
            lambda: simulated_confounding(old, estimand=_alias(old), grid=_GRID, random_state=31),
            CapabilityError,
            UNDECLARED_RULE,
        )
        assert_refused(lambda: validate_replay(old), CapabilityError, UNDECLARED_RULE)
        assert function.calls == 0, "the rule was evaluated before the refusal"

    def test_removing_the_evaluator_check_still_refuses_through_the_carry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The evaluators stop checking, and the rule runs; the refit still refuses.

        ``RegimeSet.evaluate`` and ``Rule.density`` read the module global, so the mutation
        removes both.  The frozen regime carries the source's ``None``, and ``TMLE`` holds
        its own reference to the check, so the refit meets the check of the last row.
        """
        old, function = counted_rule_fit()
        monkeypatch.setattr(base_module, "refuse_regime_densities", lambda interventions: None)
        replay = validate_replay(old)
        assert function.calls == 1
        assert [item.density_kind for item in replay.interventions] == [None]
        assert_refused(lambda: replay.refit(old.data), CapabilityError, UNDECLARED_CLASS)

    def test_a_replay_that_drops_the_declaration_refuses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation R6: without the carry, the refit of a known rule's replay refuses."""
        result = _fit_policy("rule")
        drop_the_carry(monkeypatch)
        replay = validate_replay(result)
        assert_refused(lambda: replay.refit(result.data), CapabilityError, UNDECLARED_CLASS)


# ------------------------------------------------------------------ mutation controls


def declaration_witnesses() -> list[Callable[[], None]]:
    """Every rule-declaration witness above, as a call that must raise to pass."""
    suite = TestTheRuleDeclarationIsRequired()
    return [
        suite.test_an_undeclared_rule_is_refused,
        suite.test_an_estimated_rule_is_refused_by_its_missing_term,
        lambda: suite.test_an_unknown_declaration_is_refused("Known"),
        lambda: suite.test_a_rule_that_is_not_callable_is_refused(0.6, "known"),
        TestThePositionalOrderIsUnchanged().test_the_third_positional_argument_is_the_declaration,
    ]


def evaluator_witnesses() -> list[Callable[[], None]]:
    """Every evaluator witness above, of a rule and of a user-written class."""
    suite = TestTheEvaluatorsCheckFirst()
    return [
        *(
            lambda evaluator=evaluator, name=name: (
                suite.test_a_restored_rule_refuses_before_it_runs(evaluator, name)
            )
            for evaluator in EVALUATORS
            for name in RESTORED_RULE
        ),
        *(
            lambda name=name: suite.test_an_undeclared_class_refuses_before_its_density_runs(name)
            for name in ["no attribute", *RESTORED_CLASS]
        ),
        suite.test_the_set_checks_every_regime_before_the_first_function,
    ]


def protocol_checked(value: Any) -> tuple[Any, ...]:
    """Mutation R9: ``as_interventions`` as it was, with the runtime protocol check."""
    items = list(value) if isinstance(value, (list, tuple)) else [value]
    return tuple(
        item
        if isinstance(item, Intervention) and not isinstance(item, (str, bytes))
        else Static(item)
        for item in items
    )


class TestTheWitnessesHaveTeeth:
    def test_removing_the_check_fails_every_declaration_and_evaluator_witness(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation R1: ``Rule`` and the evaluators read the module global."""
        monkeypatch.setattr(base_module, "refuse_regime_densities", lambda interventions: None)
        assert_every_witness_fails(declaration_witnesses())
        assert_every_witness_fails(evaluator_witnesses())

    @pytest.mark.parametrize("entry", list(ENTRIES))
    @pytest.mark.parametrize("name", ["rule", "no attribute", *RESTORED_CLASS])
    def test_removing_the_fit_layer_check_fails_the_fit_witnesses(
        self, entry: str, name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation R2: the regimes are evaluated after the learners, so a learner runs."""
        monkeypatch.setattr(tmle_module, "refuse_regime_densities", lambda interventions: None)
        if name == "rule":
            item, fragments = restored(spy_rule(), "rule_kind", None), (UNDECLARED_RULE,)
        else:
            item, fragments = undeclared_classes()[name]
        with pytest.raises(AssertionError):
            assert_entry_refuses(entry, item, *fragments)
        assert NeverFit.calls > 0, "the mutated fit refused before a learner"

    @pytest.mark.parametrize("entry", RECOMPUTATIONS)
    def test_removing_the_fit_layer_check_fails_the_recomputation_refusal(
        self, rule_result: Any, entry: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation R2, at a recomputation.  A refit evaluates the rule again, so its
        mutation removes the evaluator checks too."""
        old = legacy_result(rule_result)
        monkeypatch.setattr(tmle_module, "refuse_regime_densities", lambda interventions: None)
        if entry == "refit":
            monkeypatch.setattr(base_module, "refuse_regime_densities", lambda items: None)
        with pytest.raises(AssertionError):
            assert_refused(recomputations(old, RETARGETED)[entry], CapabilityError, UNDECLARED_RULE)

    @pytest.mark.parametrize("refusal", list(LATER_REFUSALS))
    @pytest.mark.parametrize("kind", list(DECLARED_AND_RESTORED))
    def test_removing_the_fit_layer_check_alone_lets_the_later_refusal_answer(
        self, refusal: str, kind: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation R2 at the order witnesses: the refusal of the configuration answers."""
        monkeypatch.setattr(tmle_module, "refuse_regime_densities", lambda interventions: None)
        suite = TestTheDeclarationRefusalComesFirst()
        with pytest.raises(AssertionError):
            suite.test_an_undeclared_object_meets_the_declaration_refusal(refusal, kind)

    def test_dropping_the_carry_fails_the_replay_witness(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation R6: the replay passes ``None``, and the refit of a known fit refuses."""
        drop_the_carry(monkeypatch)
        witness = TestTheReplay().test_a_known_rule_replays_with_the_declaration_of_each_source
        assert_every_witness_fails([witness])

    @pytest.mark.parametrize("entry", list(ENTRIES))
    def test_the_protocol_check_admits_a_bare_class_as_a_level(
        self, entry: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation R9: ``BareTilt`` falls back to ``Static``, and its refusal witness fails."""
        monkeypatch.setattr(tmle_module, "as_interventions", protocol_checked)
        assert type(protocol_checked((BareTilt(),))[0]) is Static
        suite = TestTheFitRefusesAnUndeclaredIntervention()
        with pytest.raises(AssertionError):
            suite.test_every_entry_refuses_before_any_learner_or_density_call(entry, "no attribute")
        assert NeverFit.calls > 0, "the mutated fit refused before a learner"

    def test_every_site_calls_the_one_refusal(self) -> None:
        """One refusal, one text: the fit layer calls the check that the evaluators call."""
        assert tmle_module.refuse_regime_densities is refuse_regime_densities
        assert tmle_module.as_interventions is as_interventions


# ------------------------------------------------------------------ the threshold witness


@pytest.fixture(scope="module")
def sample_rule_fit() -> Any:
    """The threshold at the sample mean, declared ``"known"``: a lie no declaration detects."""
    return threshold_fit(threshold_rule(SampleThreshold.of(threshold_frame()), rule_kind="known"))


class TestASampleThresholdRuleMisstatesTheVariance:
    def test_the_sample_threshold_is_the_fixed_threshold(self) -> None:
        """The premise: the sample mean is 0.5, and the learned rule equals the fixed one."""
        frame = threshold_frame()
        learned = SampleThreshold.of(frame)
        assert learned.centre == CENTRE
        np.testing.assert_array_equal(learned(frame), FixedThreshold()(frame))

    def test_the_fit_is_exact(self, sample_rule_fit: Any) -> None:
        """``psi = 2``, and the targeting step moves nothing: the nuisances are the truth."""
        assert sample_rule_fit.estimates["ey_regime[thr]"].psi == pytest.approx(PSI, abs=1e-12)
        for fluctuation in sample_rule_fit.fluctuations.values():
            assert np.all(np.asarray(fluctuation.epsilon) == 0.0)

    def test_control_the_curve_is_the_fixed_rule_eif(self, sample_rule_fit: Any) -> None:
        """Control: the curve is right for the functional the user declared."""
        np.testing.assert_allclose(
            threshold_curve(sample_rule_fit),
            fixed_rule_curve(threshold_frame()),
            atol=1e-10,
            rtol=0,
        )

    def test_the_regime_value_moves_with_the_threshold(self) -> None:
        """``dpsi/dc = f(c) {Q(1, c) - Q(0, c)} = 1``, by quadrature, so ``T = W - c``."""
        step = 1e-3
        slope = (regime_value(CENTRE + step) - regime_value(CENTRE - step)) / (2 * step)
        assert slope == pytest.approx(1.0, abs=1e-8)
        assert regime_value(CENTRE) == pytest.approx(PSI, abs=1e-12)

    def test_the_omitted_term_is_material_and_understates(self, sample_rule_fit: Any) -> None:
        """The nonzero witness, and the positive cross moment that makes the SE too small."""
        term = threshold_term(threshold_frame())
        assert float(np.mean(term**2)) > 0.05
        # The plan measured 0.12499375 = 1/8 - 1/(4 * 200^2).
        cross = float(np.mean(threshold_curve(sample_rule_fit) * term))
        assert cross == pytest.approx(0.125 - 1.0 / (4 * 200**2), abs=1e-12)

    def test_witness_the_reported_se_understates_the_exact_se(self, sample_rule_fit: Any) -> None:
        ratio = sample_ratio(threshold_curve(sample_rule_fit), threshold_term(threshold_frame()))
        assert ratio == pytest.approx(RATIO_PIN, abs=1e-4)
        assert ratio == pytest.approx(EXACT_RATIO, abs=3e-5)
        assert ratio < UNDERSTATEMENT_BOUND

    def test_mutation_a_frozen_threshold_oracle_loses_the_witness(
        self, sample_rule_fit: Any
    ) -> None:
        """Mutation R10: an oracle that froze the threshold has no term, reads 1, and fails."""
        curve = threshold_curve(sample_rule_fit)
        frozen = sample_ratio(curve, np.zeros_like(curve))
        assert frozen == pytest.approx(1.0, abs=1e-12)
        assert not frozen < UNDERSTATEMENT_BOUND

    def test_an_estimated_declaration_is_refused_on_this_law(self) -> None:
        """The honest declaration of the sample threshold meets the refusal, before any fit."""
        learned = SampleThreshold.of(threshold_frame())
        assert_refused(
            lambda: threshold_rule(learned, rule_kind="estimated"), CapabilityError, ESTIMATED_RULE
        )
        rule = restored(
            threshold_rule(Counter(learned), rule_kind="known"), "rule_kind", "estimated"
        )
        estimator = TMLE(
            interventions=(rule,), cross_fit=False, simultaneous=False, **never_fit_learners()
        )
        assert_refused_before_any_call(
            lambda: estimator.fit(threshold_frame(), outcome="Y", treatment="A"),
            rule.rule,
            "rule",
            ESTIMATED_RULE,
        )


class TestAKnownRuleKeepsItsInterval:
    def test_a_fixed_threshold_reports_the_same_curve_under_influence_curve(
        self, sample_rule_fit: Any
    ) -> None:
        """A stated threshold of 0.5 reports the curve of the sample threshold, bit for bit."""
        result = threshold_fit(threshold_rule(rule_kind="known"))
        estimate = result.estimates["ey_regime[thr]"]
        assert estimate.inference == "influence_curve"
        assert np.all(np.isfinite(estimate.ci))
        np.testing.assert_array_equal(threshold_curve(result), threshold_curve(sample_rule_fit))


# ------------------------------------------------------------------ the user-written witness


@pytest.fixture(scope="module")
def data_tilt_fit() -> Any:
    """The sample tilt, computed in ``density`` and declared ``"known"``."""
    return oracle_fit(TILT_COUNTS, interventions=(DataTilt(density_kind="known"),))


class TestAUserWrittenTiltUnderstatesTheVariance:
    def test_the_curve_is_the_stochastic_sample_tilt_curve(self, data_tilt_fit: Any) -> None:
        """The class admits the RM25 witness: its curve is the ``Stochastic`` one."""
        stochastic = oracle_fit(
            TILT_COUNTS,
            interventions=(
                Stochastic(SampleTilt(law.frame(TILT_COUNTS), 2.0), "tilt", density_kind="known"),
            ),
        )
        np.testing.assert_allclose(
            tilt_curve(data_tilt_fit), tilt_curve(stochastic), atol=1e-12, rtol=0
        )

    def test_witness_the_reported_se_understates_the_exact_se(self, data_tilt_fit: Any) -> None:
        ratio = se_ratio(
            tilt_curve(data_tilt_fit), law.eif("ey_ipsi[odds x2]", probs=TILT_PROBS), CELL_P
        )
        assert ratio == pytest.approx(0.6226, abs=1e-4)
        assert ratio < TILT_BOUND

    def test_mutation_a_frozen_oracle_loses_the_witness(self, data_tilt_fit: Any) -> None:
        """An oracle that froze the mechanism would read 1 and fail the bound."""
        frozen = se_ratio(tilt_curve(data_tilt_fit), fixed_eif(KnownTilt(2.0).star), CELL_P)
        assert frozen == pytest.approx(1.0, abs=1e-12)
        assert not frozen < TILT_BOUND

    def test_an_estimated_declaration_is_refused_on_this_law(self) -> None:
        """The honest declaration of the class meets the refusal, before any learner."""
        item = DataTilt(density_kind="estimated")
        estimator = TMLE(
            interventions=(item,), cross_fit=False, simultaneous=False, **never_fit_learners()
        )
        assert_refused_before_any_call(
            lambda: estimator.fit(law.frame(TILT_COUNTS), outcome="Y", treatment="A"),
            item,
            "density",
            ESTIMATED_CLASS,
            PATHWISE,
        )


class TestAKnownUserWrittenDensityKeepsItsInterval:
    def test_a_known_tilt_reports_the_fixed_density_eif_under_influence_curve(self) -> None:
        result = oracle_fit(TILT_COUNTS, interventions=(KnownUserTilt(density_kind="known"),))
        estimate = result.estimates["ey_regime[tilt]"]
        assert estimate.inference == "influence_curve"
        assert np.all(np.isfinite(estimate.ci))
        np.testing.assert_allclose(
            tilt_curve(result)[law.first_row_of(TILT_COUNTS)],
            fixed_eif(KnownTilt(2.0).star),
            atol=1e-12,
            rtol=0,
        )
