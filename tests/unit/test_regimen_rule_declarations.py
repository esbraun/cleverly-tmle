r"""A callable node of a ``DynamicRegimen`` must be declared known.

A ``DynamicRegimen`` assigns node :math:`t`'s arm by a rule :math:`d_t(H_t)`.  When every
rule is fixed before the fit, the curve that ``LTMLE`` reports is the efficient influence
function of the regimen mean.  When a rule is learned from the analysis sample, the target
changes.  A population-indexed rule, such as a threshold at the mean of :math:`W`, has a
curve with a further pathwise-derivative term, and a realized learned rule defines a
data-adaptive target.

No code can inspect a closure, so the status of the rule is the declaration
``DynamicRegimen(rule_kind=...)``, which ``Rule`` shares.  A callable written inline in a
``regimens=`` mapping carries no declaration, so a fit refuses it.  RM28 in
``docs/roadmap.md`` records the defect.  This module pins these things:

* the declaration is required when a plan holds a callable node, ``"estimated"`` is refused,
  and a plan of labels alone is exempt, with their messages;
* the positional order of ``DynamicRegimen`` is unchanged;
* an inline callable is refused, and a declared regimen in a mapping keeps its declaration;
* the check and the resolver read a plan through one helper: a ``DynamicRegimen`` stores
  an iterator plan as a tuple, and a mapping refuses an iterator plan without reading it;
* every fit entry refuses an inline or modified rule before any learner or rule call, and
  so do ``DynamicRegimen.assignment``, ``resolve_plans`` and ``resolve_regimens``;
* the declaration refusal comes before a refusal of the data or the fit configuration;
* a result whose regimen loses its declaration refuses its truncation curve before the
  replay;
* a deliberate mutation that removes a check makes those witnesses fail;
* on an exact two-node law, the regimen :math:`(1, d)` with a threshold at the sample mean,
  declared ``"known"``, gets the fixed-rule curve, which understates the standard error of
  the population-indexed target, and a fixed threshold keeps its interval.

``tests/unit/_policy_declaration_support.py`` holds the rules, the two-node law and the fit
entries.  ``tests/unit/test_rule_and_intervention_declarations.py`` tests the same
declaration on ``Rule``, and ``tests/unit/test_stochastic_regime_densities.py`` lists the
regimen among the users of the shared check.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import fields, replace
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import pytest

import cleverly.longitudinal.estimator as ltmle_module
import cleverly.longitudinal.regimen as regimen_module
from cleverly.assessment import replayability
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import CapabilityError, DataError, LongitudinalError
from cleverly.interventions import base as base_module
from cleverly.interventions.base import _ESTIMATED_RULE, _UNDECLARED_RULE
from cleverly.longitudinal import (
    LTMLE,
    DynamicRegimen,
    LongitudinalData,
    Regimen,
    resolve_plans,
    resolve_regimens,
)
from cleverly.longitudinal.estimator import (
    LONGITUDINAL_REPLAY_REGIMEN_DECLARATION,
    longitudinal_truncation_curve,
)
from cleverly.longitudinal.regimen import refuse_regimen_rules
from tests.unit._confounding_support import Counter
from tests.unit._declaration_support import (
    PANEL_COLUMNS,
    PATHWISE,
    assert_every_witness_fails,
    assert_refused,
    assert_refused_before_any_call,
    assert_replay_rows_available,
    assert_replay_rows_refused,
    modified,
    modified_states,
    never_fit_longitudinal_learners,
    panel,
    undeclared_copy,
)
from tests.unit._natural_course_support import NeverFit
from tests.unit._policy_declaration_support import (
    CENTRE,
    PSI,
    REGIMEN_EXACT_RATIO,
    REGIMEN_RATIO_PIN,
    REGIMEN_UNDERSTATEMENT_BOUND,
    RULE_UNKNOWN,
    THRESHOLD_COLUMNS,
    FixedThreshold,
    SampleThreshold,
    exact_regimen_curve,
    fixed_regimen_curve,
    longitudinal_entries,
    plugin_se,
    regimen_curve,
    regimen_fit,
    threshold_frame,
    threshold_regimen,
    threshold_term,
)

#: The two declaration refusals, imported from the module that raises them, so each text is
#: written once.  ``Rule`` meets the same two.
UNDECLARED = _UNDECLARED_RULE
ESTIMATED = _ESTIMATED_RULE
#: The remedy that the undeclared refusal names for a plan with a rule.
REMEDY = "DynamicRegimen(label, plan, rule_kind='known')"
INLINE = "A callable written inline in regimens= carries no declaration"

#: Every fit entry that can reach a regimen: ``LTMLE.fit``, ``CausalStudy.estimate`` on a
#: ``LongitudinalTreatment`` design, and ``ltmle``.  Each one fits ``panel()``.
ENTRIES = longitudinal_entries()

#: What a modified regimen can carry, and the refusal each one meets.
MODIFIED = modified_states(UNDECLARED, ESTIMATED)


def panel_rule() -> Counter:
    """``d_2(W1) = 1{W1 <= 0}`` on ``panel()``, a fixed rule that counts its calls."""
    return Counter(FixedThreshold(0.0, "W1"))


def spy_regimen() -> DynamicRegimen:
    """The known regimen ``(1, d)`` on ``panel()``, whose rule is :func:`rule_of` it."""
    return threshold_regimen(panel_rule(), rule_kind="known")


def rule_of(regimen: DynamicRegimen) -> Any:
    """The counting rule at the second node of a regimen that :func:`spy_regimen` built."""
    return regimen.plan[1]


def with_reference(regimen: Any) -> dict[str, Any]:
    """``regimens=`` as a mapping: a static reference and ``regimen`` under its own label."""
    return {"never": 0, "thr": regimen}


def panel_data() -> LongitudinalData:
    return LongitudinalData.from_frame(panel(), **PANEL_COLUMNS)


# ------------------------------------------------------------------ the declaration


class TestTheRegimenDeclarationIsRequired:
    def test_an_undeclared_regimen_is_refused(self) -> None:
        assert_refused(threshold_regimen, CapabilityError, UNDECLARED, REMEDY)

    def test_an_estimated_regimen_is_refused_by_its_missing_term(self) -> None:
        assert_refused(
            lambda: threshold_regimen(rule_kind="estimated"),
            CapabilityError,
            ESTIMATED,
            PATHWISE,
            "data-adaptive target",
        )

    @pytest.mark.parametrize("kind", ["Known", "probability", True, 1])
    def test_an_unknown_declaration_is_refused(self, kind: Any) -> None:
        assert_refused(lambda: threshold_regimen(rule_kind=kind), DataError, RULE_UNKNOWN)

    @pytest.mark.parametrize("kind", [None, "estimated"])
    def test_a_plan_of_labels_alone_is_exempt(self, kind: Any) -> None:
        """No node is a function, so no declaration is needed, and none is refused."""
        regimen = DynamicRegimen("early", (1, 0), rule_kind=kind)
        assert regimen.rule_kind == kind
        assert refuse_regimen_rules(regimen) is None
        np.testing.assert_array_equal(regimen.assignment(panel_data())[:, 0], 1)

    def test_an_unknown_declaration_is_refused_on_a_plan_of_labels(self) -> None:
        """The exemption is from the refusal, not from the check of the value."""
        assert_refused(
            lambda: DynamicRegimen("early", (1, 0), rule_kind="Known"), DataError, RULE_UNKNOWN
        )

    def test_a_known_regimen_is_accepted_and_evaluates(self) -> None:
        regimen = threshold_regimen(rule_kind="known")
        assert regimen.rule_kind == "known"
        frame = threshold_frame(2)
        assigned = regimen.assignment(LongitudinalData.from_frame(frame, **THRESHOLD_COLUMNS))
        np.testing.assert_array_equal(assigned[:, 0], 1)
        np.testing.assert_array_equal(assigned[:, 1], FixedThreshold()(frame))


class TestThePositionalOrderIsUnchanged:
    """``rule_kind`` is the last field, so ``DynamicRegimen(label, plan)`` keeps its order."""

    def test_the_declaration_is_the_last_field(self) -> None:
        assert [field.name for field in fields(DynamicRegimen)] == ["label", "plan", "rule_kind"]

    def test_the_third_positional_argument_is_the_declaration(self) -> None:
        plan = (1, FixedThreshold())
        assert DynamicRegimen("thr", plan, "known").rule_kind == "known"
        assert_refused(lambda: DynamicRegimen("thr", plan), CapabilityError, UNDECLARED)


# ------------------------------------------------------------------ inline callables


def inline_plans(rule: Any) -> dict[str, Any]:
    """Each way to write ``rule`` into ``regimens=`` without a declaration."""
    return {
        "node of a mapping plan": {"never": 0, "thr": (1, rule)},
        "rule for every node": {"never": 0, "thr": rule},
        "node of a Regimen": Regimen("thr", (1, rule)),
    }


INLINE_SHAPES = list(inline_plans(None))


class TestInlineCallablesAreRefused:
    @pytest.mark.parametrize("shape", INLINE_SHAPES)
    def test_the_check_refuses_an_inline_rule(self, shape: str) -> None:
        spec = inline_plans(FixedThreshold())[shape]
        assert_refused(lambda: refuse_regimen_rules(spec), CapabilityError, UNDECLARED, INLINE)

    @pytest.mark.parametrize("shape", INLINE_SHAPES)
    def test_resolve_regimens_refuses_an_inline_rule(self, shape: str) -> None:
        spec = inline_plans(FixedThreshold())[shape]
        assert_refused(lambda: resolve_regimens(spec, 2), CapabilityError, UNDECLARED, REMEDY)

    @pytest.mark.parametrize("entry", list(ENTRIES))
    @pytest.mark.parametrize("shape", INLINE_SHAPES)
    def test_every_entry_refuses_before_any_learner_or_rule_call(
        self, entry: str, shape: str
    ) -> None:
        rule = panel_rule()
        spec = inline_plans(rule)[shape]
        assert_refused_before_any_call(lambda: ENTRIES[entry](spec), rule, "rule", UNDECLARED)

    def test_a_static_mapping_passes(self) -> None:
        """The control: a mapping of labels needs no declaration and resolves as before."""
        spec = {"always": 1, "never": 0, "early": (1, 0)}
        assert refuse_regimen_rules(spec) is None
        assert all(type(item) is Regimen for item in resolve_regimens(spec, 2))

    def test_a_declared_regimen_in_a_mapping_keeps_its_declaration(self) -> None:
        """The key names the parameter, and the resolved regimen carries the declaration."""
        regimen = threshold_regimen(rule_kind="known")
        (resolved,) = resolve_regimens({"under a new name": regimen}, 2)
        assert type(resolved) is DynamicRegimen
        assert resolved.label == "under a new name"
        assert resolved.plan == regimen.plan
        assert resolved.rule_kind == "known"

    def test_a_declared_regimen_in_a_sequence_passes_through(self) -> None:
        (resolved,) = resolve_regimens([threshold_regimen(rule_kind="known")], 2)
        assert (resolved.label, resolved.rule_kind) == ("thr", "known")


# ------------------------------------------------------------------ the shape of a plan

#: A fragment of the refusal of a plan that is an iterator.
ITERATOR = "is an iterator"


def iterated(*nodes: Any) -> Iterator[Any]:
    """``nodes`` as a generator, which the first reader consumes."""
    return (node for node in nodes)


class TestAPlanIsReadOnce:
    """The check and the resolver read the shape of a plan through one helper.

    A ``DynamicRegimen`` stores its plan as a tuple before it checks the plan, so the check
    and every evaluator read the same nodes.  A plan in a ``regimens=`` mapping is read at
    every fit, so an iterator there is refused and not consumed.
    """

    def test_an_undeclared_iterator_plan_is_refused_at_construction(self) -> None:
        """The defect: the check skipped an iterator, and ``assignment`` ran the rule."""
        rule = panel_rule()
        assert_refused(
            lambda: DynamicRegimen("thr", iterated(1, rule)), CapabilityError, UNDECLARED
        )
        assert rule.calls == 0

    def test_a_declared_iterator_plan_keeps_its_nodes(self) -> None:
        rule = panel_rule()
        regimen = DynamicRegimen("thr", iterated(1, rule), rule_kind="known")
        assert regimen.plan == (1, rule)
        assert regimen.n_times == 2
        (resolved,) = resolve_regimens([regimen], 2)
        assert resolved.plan == (1, rule)
        np.testing.assert_array_equal(regimen.assignment(panel_data())[:, 0], 1)
        assert rule.calls == 1
        # The check reads the stored nodes, so a lost declaration still refuses the rule.
        modified(regimen, "rule_kind", None)
        assert_refused(lambda: regimen.assignment(panel_data()), CapabilityError, UNDECLARED)
        assert rule.calls == 1

    def test_a_list_plan_is_stored_as_a_tuple(self) -> None:
        """The same regimen however its plan was written, so it hashes and compares equal."""
        rule = FixedThreshold()
        regimen = DynamicRegimen("thr", [1, rule], rule_kind="known")
        assert regimen.plan == (1, rule)
        assert regimen == DynamicRegimen("thr", (1, rule), rule_kind="known")
        assert hash(regimen) == hash(DynamicRegimen("thr", (1, rule), rule_kind="known"))

    @pytest.mark.parametrize("plan", [FixedThreshold(), 1], ids=["rule", "label"])
    def test_a_single_entry_is_not_a_plan(self, plan: Any) -> None:
        assert_refused(
            lambda: DynamicRegimen("thr", plan, rule_kind="known"),
            DataError,
            "one entry per treatment node",
        )

    @pytest.mark.parametrize("rule", [True, False], ids=["with a rule", "labels alone"])
    def test_an_iterator_plan_in_a_mapping_is_refused_unread(self, rule: bool) -> None:
        plan = iterated(1, FixedThreshold() if rule else 0)
        spec = {"never": 0, "thr": plan}
        assert_refused(lambda: refuse_regimen_rules(spec), DataError, "'thr'", ITERATOR)
        assert_refused(lambda: resolve_regimens(spec, 2), DataError, "'thr'", ITERATOR)
        assert next(plan) == 1, "a check consumed the plan"

    def test_a_mapping_plan_is_refused_without_reading_its_keys_as_arms(self) -> None:
        rule = panel_rule()
        spec = {"thr": {"time one": 1, "time two": rule}}
        for evaluate in (
            lambda: refuse_regimen_rules(spec),
            lambda: resolve_regimens(spec, 2),
            lambda: ENTRIES["fit"](spec),
        ):
            assert_refused(evaluate, DataError, "regimen 'thr'", "is a mapping", "tuple")
        assert rule.calls == 0
        assert NeverFit.calls == 0

    def test_a_zero_dimensional_array_plan_is_a_data_error_before_any_learner(self) -> None:
        """The message names the sequence form and ``DynamicRegimen`` (roadmap row RM14)."""
        spec = {"thr": np.array(1)}
        fragments = (
            "regimen 'thr'",
            "zero-dimensional array",
            "a sequence with one entry per node",
            "DynamicRegimen(",
        )
        for check in (lambda: refuse_regimen_rules(spec), lambda: resolve_regimens(spec, 2)):
            assert_refused(check, DataError, *fragments)
        for entry in ENTRIES.values():
            assert_refused(partial(entry, spec), DataError, *fragments)
            assert NeverFit.calls == 0

    @pytest.mark.parametrize("entry", list(ENTRIES))
    def test_every_entry_refuses_an_iterator_plan_before_any_learner_or_rule_call(
        self, entry: str
    ) -> None:
        rule = panel_rule()
        spec = {"never": 0, "thr": iterated(1, rule)}
        assert_refused_before_any_call(
            lambda: ENTRIES[entry](spec), rule, "rule", ITERATOR, error=DataError
        )

    def test_the_iterator_refusal_comes_before_the_data_check(self) -> None:
        """``LTMLE.fit`` checks the raw ``regimens=`` before ``_prepare`` reads the frame."""
        rule = panel_rule()
        spec = {"never": 0, "thr": iterated(1, rule)}
        fit, _, _, _ = LATER_REFUSALS["missing column"]
        assert_refused_before_any_call(lambda: fit(spec), rule, "rule", ITERATOR, error=DataError)


# ------------------------------------------------------------------ the fit layer


class TestTheFitRefusesAModifiedRegimen:
    @pytest.mark.parametrize("entry", list(ENTRIES))
    @pytest.mark.parametrize("name", list(MODIFIED))
    def test_every_entry_refuses_before_any_learner_or_rule_call(
        self, entry: str, name: str
    ) -> None:
        kind, fragments = MODIFIED[name]
        regimen = modified(spy_regimen(), "rule_kind", kind)
        assert_refused_before_any_call(
            lambda: ENTRIES[entry](with_reference(regimen)), rule_of(regimen), "rule", *fragments
        )

    def test_an_unknown_declaration_is_refused_before_any_call(self) -> None:
        regimen = modified(spy_regimen(), "rule_kind", "Known")
        assert_refused_before_any_call(
            lambda: ENTRIES["fit"](with_reference(regimen)),
            rule_of(regimen),
            "rule",
            RULE_UNKNOWN,
            error=DataError,
        )

    @pytest.mark.parametrize("entry", list(ENTRIES))
    def test_a_known_declaration_reaches_the_first_learner(self, entry: str) -> None:
        """The control: the rule runs once, in ``resolve_plans``, and then a learner is fitted."""
        regimen = spy_regimen()
        with pytest.raises(AssertionError, match="before any learner is fitted"):
            ENTRIES[entry](with_reference(regimen))
        assert NeverFit.calls == 1
        assert rule_of(regimen).calls == 1


def refused_fit(
    regimens: Any,
    *,
    frame: Callable[[], pd.DataFrame] = panel,
    n_folds: int = 1,
    reference: str | None = None,
    **columns: Any,
) -> Any:
    """``LTMLE.fit`` on ``frame()`` with ``NeverFit`` learners and changed ``columns``."""
    estimator = LTMLE(
        regimens, n_folds=n_folds, reference=reference, **never_fit_longitudinal_learners()
    )
    return estimator.fit(frame(), **{**PANEL_COLUMNS, **columns})


def clustered_panel() -> pd.DataFrame:
    return panel().assign(cluster=np.repeat(np.arange(30), 2))


#: Three fits refused after the declaration check, as the fit, the class of that refusal, a
#: fragment of its message, and the rule calls the declared control makes first.  A missing
#: column is refused in ``_prepare``, before the regimens resolve.  An unknown reference and a
#: clustered cross-fit are refused after them, and the second after the rules run.
LATER_REFUSALS: dict[str, tuple[Callable[[Any], Any], type[Exception], str, int]] = {
    "missing column": (
        lambda regimens: refused_fit(regimens, baseline=["W9"]),
        DataError,
        "columns not found in the frame",
        0,
    ),
    "unknown reference": (
        lambda regimens: refused_fit(regimens, reference="missing"),
        KeyError,
        "is not one of the declared regimens",
        0,
    ),
    "clustered cross-fit": (
        lambda regimens: refused_fit(regimens, frame=clustered_panel, n_folds=2, id="cluster"),
        LongitudinalError,
        "cross-fitted longitudinal TMLE has no clustered result",
        1,
    ),
}


class TestTheDeclarationRefusalComesFirst:
    """An undeclared regimen meets its own refusal, whatever else its fit breaks.

    ``LTMLE.fit`` checks the raw ``regimens=`` before ``_prepare``.  Each later refusal
    names a remedy, and no remedy lets an undeclared rule fit.
    """

    @pytest.mark.parametrize("refusal", list(LATER_REFUSALS))
    def test_a_declared_regimen_meets_the_later_refusal(self, refusal: str) -> None:
        """The control: each configuration is refused when the declaration is intact."""
        fit, error, fragment, calls = LATER_REFUSALS[refusal]
        regimen = spy_regimen()
        assert_refused(lambda: fit(with_reference(regimen)), error, fragment)
        assert NeverFit.calls == 0
        assert rule_of(regimen).calls == calls

    @pytest.mark.parametrize("refusal", list(LATER_REFUSALS))
    def test_an_undeclared_regimen_meets_the_declaration_refusal(self, refusal: str) -> None:
        """Any exception is caught, so a later refusal that answers first fails the fragment.

        Only the declaration's ``CapabilityError`` carries the undeclared text.
        """
        fit, _, _, _ = LATER_REFUSALS[refusal]
        regimen = modified(spy_regimen(), "rule_kind", None)
        assert_refused_before_any_call(
            lambda: fit(with_reference(regimen)),
            rule_of(regimen),
            "rule",
            UNDECLARED,
            error=Exception,
        )


#: Each public evaluator that resolves or runs a regimen, called directly as a user can.
EVALUATORS: dict[str, Callable[[DynamicRegimen], Any]] = {
    "assignment": lambda regimen: regimen.assignment(panel_data()),
    "resolve_plans": lambda regimen: resolve_plans((regimen,), panel_data()),
    "resolve_regimens": lambda regimen: resolve_regimens([regimen], 2),
}


class TestTheEvaluatorsCheckFirst:
    """A modified regimen handed straight to an evaluator refuses before its rule runs."""

    @pytest.mark.parametrize("evaluator", list(EVALUATORS))
    @pytest.mark.parametrize("name", list(MODIFIED))
    def test_a_modified_regimen_refuses_before_it_runs(self, evaluator: str, name: str) -> None:
        kind, fragments = MODIFIED[name]
        regimen = modified(spy_regimen(), "rule_kind", kind)
        assert_refused(lambda: EVALUATORS[evaluator](regimen), CapabilityError, *fragments)
        assert rule_of(regimen).calls == 0, "the rule was evaluated before the refusal"

    @pytest.mark.parametrize("evaluator", list(EVALUATORS))
    def test_a_declared_regimen_is_evaluated(self, evaluator: str) -> None:
        """The control: resolving calls no rule, and each evaluation calls it once."""
        regimen = spy_regimen()
        EVALUATORS[evaluator](regimen)
        assert rule_of(regimen).calls == (0 if evaluator == "resolve_regimens" else 1)


# ------------------------------------------------------------------ a lost declaration


def undeclared_regimen(regimen: DynamicRegimen) -> DynamicRegimen:
    """``regimen`` with its declaration removed after construction."""
    return modified(regimen, "rule_kind", None)


class TestARegimenThatLosesItsDeclaration:
    def test_it_cannot_be_replaced_without_a_declaration(self) -> None:
        old = undeclared_regimen(threshold_regimen(rule_kind="known"))
        assert old.rule_kind is None
        assert_refused(lambda: replace(old), CapabilityError, UNDECLARED)
        assert replace(old, rule_kind="known").rule_kind == "known"

    def test_it_refuses_at_the_fit(self) -> None:
        old = undeclared_regimen(spy_regimen())
        assert_refused_before_any_call(
            lambda: ENTRIES["fit"](with_reference(old)), rule_of(old), "rule", UNDECLARED
        )

    def test_a_plan_of_labels_still_resolves(self) -> None:
        """The control: a plan of labels is exempt."""
        old = undeclared_regimen(DynamicRegimen("early", (1, 0), rule_kind="known"))
        assert old.rule_kind is None
        assert type(resolve_regimens([old], 2)[0]) is Regimen


def rule_regimens(result: Any) -> list[Any]:
    return [item for item in result.config.regimens if isinstance(item, DynamicRegimen)]


def undeclared(result: Any) -> Any:
    """A copy of ``result`` whose regimens no longer declare their rules known."""
    return undeclared_copy(result, "rule_kind", rule_regimens)


@pytest.fixture(scope="module")
def regimen_result() -> Any:
    """A declared regimen fit in sample on the two-node law.  Its rule is module-level."""
    return regimen_fit([threshold_regimen(rule_kind="known")])


#: The cumulative bound grid of each truncation curve here.
BOUNDS = [(0.05, 1.0)]


def forbid_the_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    """A replay after the check would be the failure, so it fails loudly."""

    def replay(*arguments: Any, **keywords: Any) -> Any:
        raise AssertionError("the recursion was replayed before the refusal")

    monkeypatch.setattr(ltmle_module, "_refit_bound", replay)


class TestAnUndeclaredLongitudinalResultRefusesARecomputation:
    """RM28: a truncation curve checks the regimens of the result as the fit does."""

    def test_the_truncation_curve_refuses_before_the_replay(
        self, regimen_result: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        old = undeclared(regimen_result)
        forbid_the_replay(monkeypatch)
        assert_refused(
            lambda: longitudinal_truncation_curve(old, BOUNDS), CapabilityError, UNDECLARED
        )

    def test_the_assessment_entry_refuses(self, regimen_result: Any) -> None:
        old = undeclared(regimen_result)
        assert not replayability(old).refit_nuisances
        capability = old.diagnostics.capability("truncation_curve")
        assert not capability.available
        assert "stored regimen rule lacks" in capability.reason
        assert_refused(
            lambda: old.diagnostics.truncation_curve(BOUNDS),
            CapabilityError,
            "stored regimen rule lacks",
        )

    def test_every_replay_row_refuses_and_a_combined_report_runs(self, regimen_result: Any) -> None:
        """A longitudinal result has no point retarget, so the rows carry the check."""
        assert_replay_rows_refused(
            undeclared(regimen_result), LONGITUDINAL_REPLAY_REGIMEN_DECLARATION
        )

    def test_a_modified_declaration_refuses(self, regimen_result: Any) -> None:
        copy = loads(dumps(regimen_result))
        modified(copy.config.regimens[0], "rule_kind", "estimated")
        assert_refused(
            lambda: longitudinal_truncation_curve(copy, BOUNDS), CapabilityError, ESTIMATED
        )

    def test_the_declared_result_recomputes(self, regimen_result: Any) -> None:
        """The control: the same curve on the result before the declaration was lost."""
        restored = loads(dumps(regimen_result))
        assert replayability(restored).refit_nuisances
        assert restored.diagnostics.capability("truncation_curve").available
        assert_replay_rows_available(restored)
        curve = longitudinal_truncation_curve(restored, BOUNDS)
        assert list(curve["estimand"]) == ["ey_regimen[thr]"]


# ------------------------------------------------------------------ the threshold witness


@pytest.fixture(scope="module")
def sample_regimen_fit() -> Any:
    """The regimen ``(1, d)`` at the sample mean, declared ``"known"``: an undetected lie."""
    learned = SampleThreshold.of(threshold_frame(2))
    return regimen_fit([threshold_regimen(learned, rule_kind="known")])


class TestASampleThresholdNodeMisstatesTheVariance:
    def test_the_sample_threshold_is_the_fixed_threshold(self) -> None:
        """The premise: the sample mean is 0.5, and the learned rule equals the fixed one."""
        frame = threshold_frame(2)
        learned = SampleThreshold.of(frame)
        assert learned.centre == CENTRE
        np.testing.assert_array_equal(learned(frame), FixedThreshold()(frame))

    def test_the_fit_is_exact(self, sample_regimen_fit: Any) -> None:
        """``psi = 2``, the plan is followed as declared, and no node's targeting moves."""
        assert sample_regimen_fit.estimates["ey_regimen[thr]"].psi == pytest.approx(PSI, abs=1e-12)
        fit = sample_regimen_fit.fits["thr"]
        np.testing.assert_array_equal(fit.assignment[:, 0], 1.0)
        np.testing.assert_array_equal(fit.assignment[:, 1], FixedThreshold()(threshold_frame(2)))
        epsilons = [np.asarray(step.fluctuation.epsilon) for step in fit.steps]
        assert len(epsilons) == 2
        assert all(np.all(epsilon == 0.0) for epsilon in epsilons)

    def test_control_the_curve_is_the_fixed_rule_eif(self, sample_regimen_fit: Any) -> None:
        """Control: the curve is right for the functional the user declared."""
        np.testing.assert_allclose(
            regimen_curve(sample_regimen_fit),
            fixed_regimen_curve(threshold_frame(2)),
            atol=1e-10,
            rtol=0,
        )

    def test_the_omitted_term_is_material_and_understates(self, sample_regimen_fit: Any) -> None:
        """The nonzero witness, and the positive cross moment that makes the SE too small."""
        term = threshold_term(threshold_frame(2))
        assert float(np.mean(term**2)) > 0.05
        # The plan measured 0.12499375 = 1/8 - 1/(4 * 200^2).
        cross = float(np.mean(regimen_curve(sample_regimen_fit) * term))
        assert cross == pytest.approx(0.125 - 1.0 / (4 * 200**2), abs=1e-12)

    def test_the_reported_se_is_the_plugin_se_of_the_curve(self, sample_regimen_fit: Any) -> None:
        """``std_error`` is ``sqrt(Var_n(curve) / n)``, so the witness below is about it."""
        reported = sample_regimen_fit.estimates["ey_regimen[thr]"].std_error
        assert reported == pytest.approx(plugin_se(regimen_curve(sample_regimen_fit)), rel=1e-12)
        # The plan measured 0.0176829.
        assert reported == pytest.approx(0.0176829, abs=1e-7)

    def test_witness_the_reported_se_understates_the_exact_se(
        self, sample_regimen_fit: Any
    ) -> None:
        """The exact SE comes from the closed forms, not from the reported curve."""
        reported = sample_regimen_fit.estimates["ey_regimen[thr]"].std_error
        ratio = reported / plugin_se(exact_regimen_curve(threshold_frame(2)))
        assert ratio == pytest.approx(REGIMEN_RATIO_PIN, abs=1e-4)
        assert ratio == pytest.approx(REGIMEN_EXACT_RATIO, abs=3e-5)
        assert ratio < REGIMEN_UNDERSTATEMENT_BOUND

    def test_mutation_a_frozen_threshold_oracle_loses_the_witness(
        self, sample_regimen_fit: Any
    ) -> None:
        """Mutation R10: an oracle that froze the threshold has no term, reads 1, and fails."""
        frozen = plugin_se(regimen_curve(sample_regimen_fit)) / plugin_se(
            fixed_regimen_curve(threshold_frame(2))
        )
        assert frozen == pytest.approx(1.0, abs=1e-12)
        assert not frozen < REGIMEN_UNDERSTATEMENT_BOUND

    def test_mutation_a_curve_with_the_term_fails_the_witness(
        self, sample_regimen_fit: Any
    ) -> None:
        """Mutation R10: the witness detects a package that adds ``T``, and one that subtracts it.

        A reported curve of ``D + T`` reads 1 and fails the bound.  ``D - T`` has second
        moment ``1/2 + 1/12 - 1/4 = 1/3``, so it reads ``sqrt(2/5)``: it passes the bound,
        and the pin fails it.
        """
        frame = threshold_frame(2)
        curve, term = regimen_curve(sample_regimen_fit), threshold_term(frame)
        exact = plugin_se(exact_regimen_curve(frame))
        right = plugin_se(curve + term) / exact
        assert right == pytest.approx(1.0, abs=1e-12)
        assert not right < REGIMEN_UNDERSTATEMENT_BOUND
        wrong = plugin_se(curve - term) / exact
        assert wrong == pytest.approx(np.sqrt(2.0 / 5.0), abs=3e-5)
        assert wrong != pytest.approx(REGIMEN_RATIO_PIN, abs=1e-4)

    def test_an_estimated_declaration_is_refused_on_this_law(self) -> None:
        """The honest declaration of the sample threshold meets the refusal, before any fit."""
        learned = SampleThreshold.of(threshold_frame(2))
        assert_refused(
            lambda: threshold_regimen(learned, rule_kind="estimated"), CapabilityError, ESTIMATED
        )
        regimen = modified(
            threshold_regimen(Counter(learned), rule_kind="known"), "rule_kind", "estimated"
        )
        estimator = LTMLE([regimen], n_folds=1, **never_fit_longitudinal_learners())
        assert_refused_before_any_call(
            lambda: estimator.fit(threshold_frame(2), **THRESHOLD_COLUMNS),
            rule_of(regimen),
            "rule",
            ESTIMATED,
            PATHWISE,
        )


class TestAKnownNodeKeepsItsInterval:
    def test_a_fixed_threshold_reports_the_same_curve_under_influence_curve(
        self, sample_regimen_fit: Any
    ) -> None:
        """A stated threshold of 0.5 reports the curve of the sample threshold, bit for bit."""
        result = regimen_fit([threshold_regimen(rule_kind="known")])
        estimate = result.estimates["ey_regimen[thr]"]
        assert estimate.inference == "influence_curve"
        assert np.all(np.isfinite(estimate.ci))
        np.testing.assert_array_equal(regimen_curve(result), regimen_curve(sample_regimen_fit))


# ------------------------------------------------------------------ mutation controls


def declaration_witnesses() -> list[Callable[[], None]]:
    """Every construction witness above, as a call that must raise to pass.

    The witnesses that call :func:`refuse_regimen_rules` by its imported name are not here:
    the mutation replaces the module global that ``DynamicRegimen`` reads, not the function.
    """
    suite = TestTheRegimenDeclarationIsRequired()
    return [
        suite.test_an_undeclared_regimen_is_refused,
        suite.test_an_estimated_regimen_is_refused_by_its_missing_term,
        lambda: suite.test_an_unknown_declaration_is_refused("Known"),
        suite.test_an_unknown_declaration_is_refused_on_a_plan_of_labels,
        TestThePositionalOrderIsUnchanged().test_the_third_positional_argument_is_the_declaration,
    ]


def evaluator_witnesses() -> list[Callable[[], None]]:
    """Every evaluator witness above, and the refusal of an inline rule by the resolver."""
    suite = TestTheEvaluatorsCheckFirst()
    inline = TestInlineCallablesAreRefused()
    return [
        *(
            lambda evaluator=evaluator, name=name: (
                suite.test_a_modified_regimen_refuses_before_it_runs(evaluator, name)
            )
            for evaluator in EVALUATORS
            for name in MODIFIED
        ),
        *(
            lambda shape=shape: inline.test_resolve_regimens_refuses_an_inline_rule(shape)
            for shape in INLINE_SHAPES
        ),
    ]


def remove_the_regimen_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation R3: ``DynamicRegimen`` and the evaluators read the module global."""
    monkeypatch.setattr(regimen_module, "refuse_regimen_rules", lambda regimens: None)


def remove_the_fit_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation R4: ``LTMLE.fit`` and the truncation curve hold their own reference."""
    monkeypatch.setattr(ltmle_module, "refuse_regimen_rules", lambda regimens: None)


def drop_the_resolve_carry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation R5: ``_resolve_one`` rebuilds a ``DynamicRegimen`` without its ``rule_kind``."""
    original = regimen_module._resolve_one

    def dropping(label: str, plan: Any, n_times: int) -> Any:
        return original(label, plan.plan if isinstance(plan, DynamicRegimen) else plan, n_times)

    monkeypatch.setattr(regimen_module, "_resolve_one", dropping)


def fit_witness(entry: str, name: str) -> None:
    """One fit witness: a modified regimen, or an inline rule, through one entry."""
    suite = TestTheFitRefusesAModifiedRegimen()
    if name in MODIFIED:
        suite.test_every_entry_refuses_before_any_learner_or_rule_call(entry, name)
    else:
        TestInlineCallablesAreRefused().test_every_entry_refuses_before_any_learner_or_rule_call(
            entry, name
        )


class TestTheWitnessesHaveTeeth:
    def test_removing_the_regimen_check_fails_every_declaration_and_evaluator_witness(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation R3: the fit keeps its own check, so only these witnesses fail."""
        remove_the_regimen_checks(monkeypatch)
        assert_every_witness_fails(declaration_witnesses())
        assert_every_witness_fails(evaluator_witnesses())
        assert threshold_regimen().rule_kind is None

    def test_removing_the_fit_check_alone_lets_a_data_refusal_answer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation R4: ``_prepare`` refuses the missing column before the regimens resolve.

        The unknown reference and the clustered cross-fit are refused after
        ``resolve_regimens``, which rebuilds the modified regimen and refuses it, so R4
        alone leaves those two witnesses passing.
        """
        remove_the_fit_check(monkeypatch)
        suite = TestTheDeclarationRefusalComesFirst()
        assert_every_witness_fails(
            [
                lambda: suite.test_an_undeclared_regimen_meets_the_declaration_refusal(
                    "missing column"
                )
            ]
        )
        suite.test_an_undeclared_regimen_meets_the_declaration_refusal("unknown reference")
        suite.test_an_undeclared_regimen_meets_the_declaration_refusal("clustered cross-fit")

    def test_removing_the_fit_check_fails_the_undeclared_truncation_witness(
        self, regimen_result: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation R4: the replay runs from the saved plans, which no rule check guards."""
        remove_the_fit_check(monkeypatch)
        suite = TestAnUndeclaredLongitudinalResultRefusesARecomputation()
        with pytest.raises(AssertionError, match="replayed before the refusal"):
            suite.test_the_truncation_curve_refuses_before_the_replay(regimen_result, monkeypatch)

    @pytest.mark.parametrize("refusal", list(LATER_REFUSALS))
    def test_removing_both_checks_lets_every_later_refusal_answer(
        self, refusal: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutations R4 and R3 together: no check refuses the rule, so the later one answers."""
        remove_the_fit_check(monkeypatch)
        remove_the_regimen_checks(monkeypatch)
        suite = TestTheDeclarationRefusalComesFirst()
        assert_every_witness_fails(
            [lambda: suite.test_an_undeclared_regimen_meets_the_declaration_refusal(refusal)]
        )

    @pytest.mark.parametrize("entry", list(ENTRIES))
    @pytest.mark.parametrize("name", [*MODIFIED, INLINE_SHAPES[0]])
    def test_removing_both_checks_fails_the_fit_witnesses(
        self, entry: str, name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutations R4 and R3 together: the rule runs and a learner is fitted."""
        remove_the_fit_check(monkeypatch)
        remove_the_regimen_checks(monkeypatch)
        with pytest.raises(AssertionError):
            fit_witness(entry, name)
        assert NeverFit.calls > 0, "the mutated fit refused before a learner"

    @pytest.mark.parametrize("entry", list(ENTRIES))
    def test_removing_the_fit_check_alone_still_refuses_before_any_learner(
        self, entry: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation R4 alone: ``resolve_regimens`` rebuilds the regimen, which refuses."""
        remove_the_fit_check(monkeypatch)
        for name in [*MODIFIED, INLINE_SHAPES[0]]:
            fit_witness(entry, name)

    def test_dropping_the_resolve_carry_fails_the_declared_mapping_witness(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation R5: the rebuilt regimen reads ``None``, so resolving it refuses."""
        drop_the_resolve_carry(monkeypatch)
        witness = TestInlineCallablesAreRefused().test_a_declared_regimen_in_a_mapping_keeps_its_declaration
        with pytest.raises((AssertionError, CapabilityError)):
            witness()

    def test_every_site_calls_the_one_refusal(self) -> None:
        """One refusal, one text: the fit calls the check that the evaluators call, and the
        regimen shares the declaration of ``Rule``."""
        assert ltmle_module.refuse_regimen_rules is refuse_regimen_rules
        assert regimen_module._RULE_DECLARATION is base_module._RULE_DECLARATION
