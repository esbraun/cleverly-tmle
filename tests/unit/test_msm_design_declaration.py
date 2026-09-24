r"""An MSM design must be declared known, and an estimated one is refused.

The working design :math:`\varphi(a, V)` is part of the estimand.  When it is a fixed
function, the influence curve the package reports is the efficient influence function of
:math:`\beta`.  When it is computed from the sample, such as a covariate centred at its sample
mean, :math:`\varphi` is a functional of :math:`P`.  The efficient influence function then
carries a further term for the pathwise derivative through that statistic, and the reported
curve does not have it.

A callable can close over any estimate, and no code can inspect a closure, so the status of
the design is a declaration: ``MSM(design_kind=...)``.  ``MSM.linear`` writes none: the
exact type of the design it builds reads as known.  RM27 in ``docs/roadmap.md`` records the
defect.  This module pins these things:

* the declaration is required, ``"estimated"`` is refused, and a design that is not callable
  is refused, with their messages;
* ``design_kind=None`` reads as known only on a design that ``MSM.linear`` built, by its
  exact type, so a user design swapped into the shorthand needs its own declaration, and a
  shorthand pickled before the field existed still reads as known;
* every fit entry refuses a restored or modified model before any learner or design call,
  and so do ``MSMSet.evaluate`` and ``evaluate_regimen_msm`` called directly;
* a restored model meets its declaration refusal before a refusal of the fit configuration;
* a result restored with an undeclared written design keeps its stored estimates, and every
  recomputation from it refuses, while a restored ``MSM.linear`` result still recomputes;
* the simulated-confounding replay refuses a result restored without either MSM
  declaration before it runs the design or the weight, and carries the declaration of a
  model it admits;
* a deliberate mutation that removes a refusal makes those witnesses fail;
* on an exact law, a design centred at the sample mean of ``W`` and declared ``"known"`` gets
  the fixed-centre curve, which understates the standard error of the intercept;
* a design with a fixed centre, and ``MSM.linear``, keep their intervals.

``tests/unit/test_msm_projection_weights.py`` holds the other tests of the weight
declaration, and ``tests/unit/_msm_declaration_support.py`` holds the builders the two
files share.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

import cleverly.longitudinal.estimator as ltmle_module
import cleverly.longitudinal.msm as regimen_msm_module
import cleverly.msm as msm_module
from cleverly import MSMProjection
from cleverly.assessment import replayability
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError, DataError
from cleverly.msm import (
    _ESTIMATED_DESIGN,
    _UNDECLARED_DESIGN,
    _UNDECLARED_WEIGHTS,
    MSM,
    MSMSet,
    _LinearDesign,
    refuse_msm_functions,
)
from cleverly.sensitivity import _simulated_confounding_fixed as replay_module
from cleverly.sensitivity import simulated_confounding
from tests import discrete_law as law
from tests.pickles import legacy_without
from tests.unit._confounding_support import Counter, forbid_draw_and_refit, validate_replay
from tests.unit._declaration_support import (
    PATHWISE,
    assert_every_witness_fails,
    assert_keeps_its_interval,
    assert_refused,
    assert_refused_before_any_call,
    assert_stored_interval_is_a_diagnostic,
    cell_p,
    oracle_fit,
    recomputations,
    restored,
    restored_states,
    se_ratio,
    tmle_module,
)
from tests.unit._msm_declaration_support import (
    DESIGN_UNKNOWN,
    DOSE_SLOPE,
    MSM_ENTRIES,
    CentredDesign,
    dose_surface,
    drop_from_the_replay,
    duration_design,
    evaluate_point,
    evaluate_regimen,
    in_sample_fit,
    legacy_msm_result,
    linear,
    ltmle_fit,
    msm_curve,
    msm_eif,
    remove_every_fit_check,
    tmle_fit,
    uniform_dose_fit,
    written,
)
from tests.unit._natural_course_support import NeverFit, never_fit_learners
from tests.unit._simulated_confounding_support import (
    _GRID,
    _alias,
    _estimate,
    _fit_msm,
    _MSMDesign,
    _MSMWeight,
    _study,
)

#: The two declaration refusals, imported from the module that raises them, so the text
#: is written once.
UNDECLARED = _UNDECLARED_DESIGN
ESTIMATED = _ESTIMATED_DESIGN
#: Fragments of the other refusals.  A test matches a fragment, not the whole text, so a
#: rewording of the explanation does not break it, but a message from another check does.
SIGNATURES = (
    "MSM design= must be a callable",
    "(arm_label, covariate_frame) -> (n, p) for a point treatment",
    "(regimen_label, horizon, baseline_frame) -> (n, p) for a longitudinal regimen MSM",
)


def full_rank_design(*arguments: Any) -> np.ndarray:
    """A known design, ``[1, a]`` for an arm and ``[1, duration]`` for a regimen.

    It is full rank, because a valid fit evaluates the design and checks its rank before the
    first learner.  A ``Counter`` around it counts its calls.
    """
    if len(arguments) == 3:
        return duration_design(*arguments)
    arm, frame = arguments
    return np.column_stack([np.ones(len(frame)), np.full(len(frame), float(arm))])


# ------------------------------------------------------------------ the declaration


class TestTheDeclarationIsRequired:
    def test_an_undeclared_design_is_refused(self) -> None:
        assert_refused(written, CapabilityError, UNDECLARED, "design_kind='known'")

    def test_an_estimated_design_is_refused_by_its_missing_term(self) -> None:
        assert_refused(
            lambda: written(design_kind="estimated"),
            CapabilityError,
            ESTIMATED,
            PATHWISE,
            "functional of P",
        )

    @pytest.mark.parametrize("kind", ["Known", "probability", True, 1])
    def test_an_unknown_declaration_is_refused(self, kind: Any) -> None:
        assert_refused(lambda: written(design_kind=kind), DataError, DESIGN_UNKNOWN)

    @pytest.mark.parametrize("design", [np.ones((3, 3)), 0.5], ids=["array", "float"])
    @pytest.mark.parametrize("kind", [None, "known", "estimated"])
    def test_a_design_that_is_not_callable_is_refused(self, design: Any, kind: Any) -> None:
        """The callable check runs first, so every declaration meets the same refusal."""
        assert_refused(lambda: written(design, design_kind=kind), DataError, *SIGNATURES)

    def test_a_known_design_is_accepted_for_both_signatures(self) -> None:
        assert written(design_kind="known").design_kind == "known"
        regimen = MSM(
            design=duration_design, terms=("(intercept)", "duration"), design_kind="known"
        )
        assert regimen.design_kind == "known"

    def test_the_shorthand_design_is_known_by_its_type(self) -> None:
        """``MSM.linear`` writes no declaration: the exact type of its design reads known."""
        for model in (linear(), MSM.linear()):
            assert model.design_kind is None
            assert msm_module._design_kind(model) == "known"
            assert refuse_msm_functions(model) is None


# ------------------------------------------------------------------ the exact-type rule


class TestTheExactTypeRuleReadsTheShorthand:
    """``design_kind=None`` reads known only on a design that ``MSM.linear`` built.

    The shorthand and a model saved before the field existed both carry ``None``, so the
    one rule serves both, and nothing else reads as known without a declaration.
    """

    def test_replacing_the_shorthand_design_drops_its_declaration(self) -> None:
        """A user design swapped into ``MSM.linear`` inherits no declaration from it."""
        assert_refused(
            lambda: replace(linear(), design=CentredDesign()), CapabilityError, UNDECLARED
        )
        assert_refused(
            lambda: replace(linear(), design_kind="estimated"), CapabilityError, ESTIMATED, PATHWISE
        )
        known = replace(linear(), design=CentredDesign(), design_kind="known")
        assert known.design_kind == "known"

    def test_a_shorthand_pickled_before_the_field_reads_known(self) -> None:
        old = legacy_without(linear(), "design_kind")
        assert "design_kind" not in vars(old)
        assert old.design_kind is None
        assert refuse_msm_functions(old) is None
        assert replace(old).design_kind is None

    def test_a_legacy_written_design_refuses(self) -> None:
        old = legacy_without(written(design_kind="known"), "design_kind")
        assert old.design_kind is None
        assert_refused(lambda: replace(old), CapabilityError, UNDECLARED)
        assert replace(old, design_kind="known").design_kind == "known"

    def test_a_forged_shorthand_flag_is_not_a_declaration(self) -> None:
        """``from_linear=True`` can be set on any design, so the rule does not read it."""
        assert_refused(lambda: written(from_linear=True), CapabilityError, UNDECLARED)

    def test_a_subclass_of_the_shorthand_design_is_not_a_declaration(self) -> None:
        """A subclass passes ``isinstance``, so the rule tests the exact type."""

        class Subclass(_LinearDesign):
            pass

        assert_refused(
            lambda: MSM(design=Subclass(("W",), False), terms=law.MSM_TERMS),
            CapabilityError,
            UNDECLARED,
        )


# ------------------------------------------------------------------ the fit layer


def point_model() -> MSM:
    """A point MSM whose known design counts its calls in ``model.design.calls``."""
    return MSM(design=Counter(full_rank_design), terms=("(intercept)", "a"), design_kind="known")


def regimen_model() -> MSM:
    """A regimen MSM whose known design counts its calls in ``model.design.calls``."""
    return MSM(
        design=Counter(full_rank_design), terms=("(intercept)", "duration"), design_kind="known"
    )


def without_learners(fit: Callable[[MSM, dict[str, Any]], Any]) -> Callable[[MSM], Any]:
    """``fit`` with fresh ``NeverFit`` learners, which reset ``NeverFit.calls``."""
    return lambda model: fit(model, never_fit_learners())


#: Every fit entry that can reach a restored model: its model builder and the fit.  Each
#: fit resets ``NeverFit``.
ENTRIES: dict[str, tuple[Callable[[], MSM], Callable[[MSM], Any]]] = {
    **{name: (point_model, without_learners(fit)) for name, fit in MSM_ENTRIES.items()},
    "ltmle": (regimen_model, ltmle_fit),
}

#: What a restored model can carry, and the refusal each one meets.
RESTORED = restored_states(UNDECLARED, ESTIMATED)

#: What a restored model can carry that is not a declaration, as the field, its value, and
#: the fragments of the ``DataError`` it meets.
MALFORMED: dict[str, tuple[str, Any, tuple[str, ...]]] = {
    "unknown": ("design_kind", "Known", (DESIGN_UNKNOWN,)),
    "not callable": ("design", np.ones((3, 2)), SIGNATURES),
}


def assert_entry_refuses(entry: str, kind: Any, *fragments: str) -> None:
    build, fit = ENTRIES[entry]
    model = restored(build(), "design_kind", kind)
    assert_refused_before_any_call(lambda: fit(model), model.design, "design", *fragments)


class TestTheFitRefusesARestoredModel:
    @pytest.mark.parametrize("entry", list(ENTRIES))
    @pytest.mark.parametrize("name", list(RESTORED))
    def test_every_entry_refuses_before_any_learner_or_design_call(
        self, entry: str, name: str
    ) -> None:
        kind, fragments = RESTORED[name]
        assert_entry_refuses(entry, kind, *fragments)

    @pytest.mark.parametrize("entry", list(ENTRIES))
    @pytest.mark.parametrize("name", list(MALFORMED))
    def test_every_entry_refuses_a_malformed_model_before_any_call(
        self, entry: str, name: str
    ) -> None:
        """An unknown declaration and a design that is not callable are data errors."""
        field, value, fragments = MALFORMED[name]
        build, fit = ENTRIES[entry]
        model = build()
        spy = model.design
        restored(model, field, value)
        assert_refused_before_any_call(
            lambda: fit(model), spy, "design", *fragments, error=DataError
        )

    @pytest.mark.parametrize("entry", list(ENTRIES))
    def test_a_known_declaration_reaches_the_first_learner(self, entry: str) -> None:
        """The control: the same entries with the declaration intact go past the check."""
        build, fit = ENTRIES[entry]
        model = build()
        with pytest.raises(AssertionError, match="before any learner is fitted"):
            fit(model)
        assert NeverFit.calls == 1


def cross_fitted(model: MSM, frame: Any, **settings: Any) -> Any:
    """A cross-fitted ``TMLE.fit`` of ``model`` on ``frame``, with ``NeverFit`` learners.

    ``settings`` holds the ``TMLE`` keywords, and ``delta`` goes to the fit.
    """
    delta = settings.pop("delta", None)
    estimator = TMLE(
        msm=model, cross_fit=True, n_folds=2, simultaneous=False, **never_fit_learners(), **settings
    )
    return estimator.fit(frame, outcome="Y", treatment="A", delta=delta)


def continuous_outcome() -> Any:
    """The default law with the outcome ``Y + W / 2``, which reads as continuous."""
    frame = law.frame()
    return frame.assign(Y=frame["Y"] + 0.5 * frame["W"])


def missing_outcome() -> Any:
    """The default law with every seventh outcome missing, and ``D`` its indicator."""
    frame = law.frame()
    observed = np.arange(len(frame)) % 7 != 0
    return frame.assign(D=observed.astype(float), Y=frame["Y"].where(observed))


#: Three fits that a later refusal stops before ``MSMSet.evaluate`` runs, as the fit, the
#: class of that refusal, and a fragment of its message.  ``_resolve_estimands_for_data``
#: raises the scale and missing-outcome refusals after its model check, and ``_fit_single``
#: raises the ``cv_evaluation`` refusal after that method returns.
LATER_REFUSALS: dict[str, tuple[Callable[[MSM], Any], type[Exception], str]] = {
    "cv_evaluation": (
        lambda model: cross_fitted(model, law.frame(), cv_evaluation=True),
        ValueError,
        "cv_evaluation=True does not yet support ['msm']",
    ),
    "unbounded scale": (
        lambda model: cross_fitted(model, continuous_outcome()),
        CapabilityError,
        "needs a declared q_bounds",
    ),
    "missing outcomes": (
        lambda model: cross_fitted(model, missing_outcome(), delta="D"),
        CapabilityError,
        "no audited result covers MSM targets under cross-fitting",
    ),
}


class TestTheDeclarationRefusalComesFirst:
    """A restored model meets its own refusal, whatever else its fit configuration breaks.

    ``_resolve_estimands_for_data`` checks the model before any refusal of the fit
    configuration.  Each of those refusals names a remedy, and no remedy lets an undeclared
    or estimated design fit.  Each fit that no such refusal stops reaches
    ``MSMSet.evaluate``, which runs the same check before the first learner.  So these
    tests are the witness of the model check in ``_resolve_estimands_for_data``.
    """

    @pytest.mark.parametrize("refusal", list(LATER_REFUSALS))
    def test_a_declared_model_meets_the_later_refusal(self, refusal: str) -> None:
        """The control: each configuration is refused when the declaration is intact."""
        fit, error, fragment = LATER_REFUSALS[refusal]
        model = point_model()
        assert_refused_before_any_call(
            lambda: fit(model), model.design, "design", fragment, error=error
        )

    @pytest.mark.parametrize("refusal", list(LATER_REFUSALS))
    @pytest.mark.parametrize("name", list(RESTORED))
    def test_a_restored_model_meets_the_declaration_refusal(self, refusal: str, name: str) -> None:
        kind, fragments = RESTORED[name]
        fit, _, _ = LATER_REFUSALS[refusal]
        model = restored(point_model(), "design_kind", kind)
        assert_refused_before_any_call(lambda: fit(model), model.design, "design", *fragments)


#: Each public evaluator that runs the design, with a builder of a model it accepts.
EVALUATORS: dict[str, tuple[Callable[[], MSM], Callable[[MSM], Any]]] = {
    "MSMSet.evaluate": (point_model, evaluate_point),
    "evaluate_regimen_msm": (regimen_model, evaluate_regimen),
}


class TestTheEvaluatorsCheckFirst:
    """A restored model handed straight to an evaluator refuses before its design runs."""

    @pytest.mark.parametrize("evaluator", list(EVALUATORS))
    @pytest.mark.parametrize("name", list(RESTORED))
    def test_a_restored_model_refuses_before_the_design_runs(
        self, evaluator: str, name: str
    ) -> None:
        kind, fragments = RESTORED[name]
        build, evaluate = EVALUATORS[evaluator]
        model = restored(build(), "design_kind", kind)
        assert_refused(lambda: evaluate(model), CapabilityError, *fragments)
        assert model.design.calls == 0, "the design was evaluated before the refusal"

    @pytest.mark.parametrize("evaluator", list(EVALUATORS))
    def test_a_declared_model_is_evaluated(self, evaluator: str) -> None:
        """The control: the same call with the declaration intact runs the design."""
        build, evaluate = EVALUATORS[evaluator]
        model = build()
        evaluate(model)
        assert model.design.calls > 0


# ------------------------------------------------------------------ a restored result


def legacy_result(result: Any) -> Any:
    """``result`` as an artifact written before ``design_kind`` existed would restore it."""
    return legacy_msm_result(result, "design_kind")


#: The estimands a retarget of the legacy result requests.
RETARGETED = ("msm",)
RECOMPUTATIONS = ["truncation_curve", "retarget", "refit"]


class TestALegacyResultKeepsItsPointEstimatesAndRefusesARecomputation:
    """RM27: a restored result with a written design keeps its point estimates.

    Loading raises nothing.  A written design restored without its declaration gives the
    result the ``"undeclared_function_plugin"`` status of RM28, so the stored interval
    becomes a diagnostic.  Every sweep
    recomputes through ``_retarget_detailed``, and a refit through
    ``_resolve_estimands_for_data``, and both check the declaration as the fit does.
    """

    @pytest.fixture(scope="class")
    def result(self) -> Any:
        return in_sample_fit(written(design_kind="known"))

    @pytest.fixture(scope="class")
    def shorthand(self) -> Any:
        return in_sample_fit(linear())

    def test_the_stored_interval_becomes_a_diagnostic(self, result: Any) -> None:
        assert "msm[W]" in result.estimates
        old = legacy_result(result)
        assert_stored_interval_is_a_diagnostic(result, old)
        assert not replayability(old).retarget_cached_nuisances
        assert not replayability(old).refit_nuisances
        assert not old.diagnostics.capability("truncation_curve").available
        assert replayability(result).refit_nuisances

    def test_a_legacy_shorthand_result_keeps_its_interval(self, shorthand: Any) -> None:
        """The over-refusal control: ``MSM.linear`` saved before both declarations existed."""
        old = legacy_msm_result(legacy_result(shorthand), "weights_kind")
        assert type(old.estimator.msm.design) is _LinearDesign
        assert_keeps_its_interval(shorthand, old)

    @pytest.mark.parametrize("entry", RECOMPUTATIONS)
    def test_every_recomputation_refuses(self, result: Any, entry: str) -> None:
        old = legacy_result(result)
        assert_refused(recomputations(old, RETARGETED)[entry], CapabilityError, UNDECLARED)

    @pytest.mark.parametrize("entry", RECOMPUTATIONS)
    def test_the_declared_result_recomputes(self, result: Any, entry: str) -> None:
        """The control: the same entry on the result before the declaration was lost."""
        recomputations(result, RETARGETED)[entry]()

    @pytest.mark.parametrize("entry", RECOMPUTATIONS)
    def test_a_legacy_shorthand_result_still_recomputes(self, shorthand: Any, entry: str) -> None:
        """The over-refusal control: a result saved before both MSM declarations existed."""
        old = legacy_msm_result(legacy_result(shorthand), "weights_kind")
        assert type(old.estimator.msm.design) is _LinearDesign
        recomputations(old, RETARGETED)[entry]()


# ------------------------------------------------------------------ the replay


def shorthand_fit() -> Any:
    """A replayable fit of ``MSM.linear``, with uniform weights."""
    return _estimate(_study(), MSMProjection(linear()))


def legacy_shorthand_fit() -> Any:
    return legacy_result(shorthand_fit())


#: The coefficient of the shorthand fit that the replay targets.
SLOPE = "a"

#: Each declaration that a replayed model can lack, and the refusal it meets.
UNDECLARED_BY_FIELD = {"weights_kind": _UNDECLARED_WEIGHTS, "design_kind": _UNDECLARED_DESIGN}


def counted_msm_fit(field: str) -> tuple[Any, Counter, Counter]:
    """A fitted MSM whose design and weight count their calls, restored without ``field``.

    Both functions are declared known before the fit, so the restored model lacks only the
    declaration ``field`` names.  The counters are read off the restored result, because a
    pickle round trip copies them.
    """
    model = MSM(
        design=Counter(_MSMDesign()),
        terms=("intercept", "treatment", "baseline"),
        weights=Counter(_MSMWeight()),
        weights_kind="known",
        design_kind="known",
    )
    old = legacy_msm_result(_estimate(_study(), MSMProjection(model)), field)
    design, weights = old.estimator.msm.design, old.estimator.msm.weights
    design.calls = weights.calls = 0
    return old, design, weights


def remove_the_evaluator_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation M3: ``MSMSet.evaluate`` runs the user functions without checking the model.

    ``MSM.__post_init__`` reads the same module global, so the no-op holds only while
    ``evaluate`` runs, and the ``replace`` that builds a replay model still checks.
    """
    evaluate = MSMSet.evaluate.__func__

    def unchecked(cls: type[MSMSet], msm: MSM, data: Any) -> MSMSet:
        with pytest.MonkeyPatch.context() as inner:
            inner.setattr(msm_module, "refuse_msm_functions", lambda model: None)
            return evaluate(cls, msm, data)

    monkeypatch.setattr(MSMSet, "evaluate", classmethod(unchecked))


class TestTheReplayChecksAndCarriesTheDeclaration:
    @pytest.mark.parametrize("field", list(UNDECLARED_BY_FIELD))
    def test_a_legacy_declaration_refuses_before_any_user_function_runs(
        self, field: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A result restored without either declaration refuses before any user function."""
        old, design, weights = counted_msm_fit(field)
        forbid_draw_and_refit(monkeypatch, old.estimator)
        assert_refused(
            lambda: simulated_confounding(old, estimand=_alias(old), grid=_GRID, random_state=31),
            CapabilityError,
            UNDECLARED_BY_FIELD[field],
        )
        assert_refused(lambda: validate_replay(old), CapabilityError, UNDECLARED_BY_FIELD[field])
        assert design.calls == 0, "the design was evaluated before the refusal"
        assert weights.calls == 0, "the weight was evaluated before the refusal"

    @pytest.mark.parametrize("field", list(UNDECLARED_BY_FIELD))
    def test_removing_the_evaluator_check_evaluates_both_and_still_refuses(
        self, field: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation M3: without the check in ``MSMSet.evaluate`` the replay runs both.

        ``replace`` still refuses the undeclared function, because the replay passes the
        source model's declaration and never forges ``"known"``.  The refusal arrives after
        the user functions ran, which is what the check removes.
        """
        old, design, weights = counted_msm_fit(field)
        remove_the_evaluator_check(monkeypatch)
        assert_refused(lambda: validate_replay(old), CapabilityError, UNDECLARED_BY_FIELD[field])
        assert design.calls == 2
        assert weights.calls == 2

    def test_a_known_written_design_replays_declared(self) -> None:
        assert validate_replay(_fit_msm()).msm.design_kind == "known"

    def test_a_legacy_shorthand_fit_replays(self) -> None:
        """The frozen arrays replace ``_LinearDesign``, so the replay carries the rule.

        The shorthand model holds ``None`` whether it was fitted now or restored from an
        artifact that predates the field, so both replay declared ``"known"``.
        """
        result = shorthand_fit()
        assert result.estimator.msm.design_kind is None
        assert validate_replay(result, SLOPE).msm.design_kind == "known"
        alias = _alias(result, coefficient=SLOPE)
        expected = simulated_confounding(result, estimand=alias, grid=_GRID, random_state=31)
        old = legacy_shorthand_fit()
        assert "design_kind" not in vars(old.estimator.msm)
        surface = simulated_confounding(old, estimand=alias, grid=_GRID, random_state=31)
        assert all(cell.failure is None for cell in surface.cells)
        assert surface == expected
        assert validate_replay(old, SLOPE).msm.design_kind == "known"

    def test_a_legacy_shorthand_dose_fit_replays(self) -> None:
        """The continuous twin: the frozen dose functions replace ``_LinearDesign`` too.

        The continuous replay builds its model with its own ``replace`` call, so the
        discrete witness above does not cover it.
        """
        result = uniform_dose_fit()
        old = legacy_result(result)
        assert "design_kind" not in vars(old.estimator.msm)
        assert dose_surface(old) == dose_surface(result)
        assert validate_replay(old, DOSE_SLOPE).msm.design_kind == "known"

    def test_a_replay_that_drops_the_declaration_refuses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation M4: drop what ``_freeze_msm`` passes.

        A written design declares ``"known"``, which ``replace`` would carry on its own, so
        the witness is a shorthand fit.  Its model holds ``None``, so only the replay
        supplies the declaration.  The RM13 file holds the same test for the weight.
        """
        result = shorthand_fit()
        drop_from_the_replay(monkeypatch, "design_kind")
        assert_refused(lambda: validate_replay(result, SLOPE), CapabilityError, UNDECLARED)


# ------------------------------------------------------------------ mutation controls


def declaration_witnesses() -> list[Callable[[], None]]:
    """Every declaration-layer witness above, as a call that must raise to pass."""
    suite = TestTheDeclarationIsRequired()
    rule = TestTheExactTypeRuleReadsTheShorthand()
    return [
        suite.test_an_undeclared_design_is_refused,
        suite.test_an_estimated_design_is_refused_by_its_missing_term,
        lambda: suite.test_an_unknown_declaration_is_refused("Known"),
        lambda: suite.test_a_design_that_is_not_callable_is_refused(np.ones((3, 3)), "known"),
        rule.test_replacing_the_shorthand_design_drops_its_declaration,
        rule.test_a_forged_shorthand_flag_is_not_a_declaration,
        rule.test_a_subclass_of_the_shorthand_design_is_not_a_declaration,
    ]


class TestTheWitnessesHaveTeeth:
    def test_removing_the_declaration_check_fails_every_declaration_witness(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation M1."""
        monkeypatch.setattr(msm_module, "refuse_msm_functions", lambda model: None)
        assert_every_witness_fails(declaration_witnesses())

    @pytest.mark.parametrize("entry", list(ENTRIES))
    @pytest.mark.parametrize("name", list(RESTORED))
    def test_removing_the_fit_layer_check_fails_the_fit_witnesses(
        self, entry: str, name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation M2, at the fit, which removes the evaluator checks too."""
        kind, fragments = RESTORED[name]
        remove_every_fit_check(monkeypatch)
        with pytest.raises(AssertionError):
            assert_entry_refuses(entry, kind, *fragments)
        assert NeverFit.calls > 0, "the mutated fit refused before a learner"

    @pytest.mark.parametrize("entry", RECOMPUTATIONS)
    def test_removing_the_fit_layer_check_fails_the_recomputation_refusal(
        self, entry: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation M2, at a recomputation of a legacy result.

        A refit evaluates the design again, so its mutation removes the evaluator checks
        too.  A sweep and a retarget reuse the stored arrays, so the check in
        ``_retarget_detailed`` is the only one on their path.
        """
        old = legacy_result(in_sample_fit(written(design_kind="known")))
        if entry == "refit":
            remove_every_fit_check(monkeypatch)
        else:
            monkeypatch.setattr(tmle_module, "refuse_msm_functions", lambda model: None)
        with pytest.raises(AssertionError):
            assert_refused(recomputations(old, RETARGETED)[entry], CapabilityError, UNDECLARED)

    @pytest.mark.parametrize("refusal", list(LATER_REFUSALS))
    def test_removing_the_fit_layer_check_alone_lets_the_later_refusal_answer(
        self, refusal: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation M7: the fit layer stops checking, and the evaluator checks stay.

        No learner and no design runs either way, so only the refusal that answers shows
        the mutation.  The restored model meets the refusal of the configuration.
        """
        fit, error, fragment = LATER_REFUSALS[refusal]
        monkeypatch.setattr(tmle_module, "refuse_msm_functions", lambda model: None)
        model = restored(point_model(), "design_kind", None)
        assert_refused_before_any_call(
            lambda: fit(model), model.design, "design", fragment, error=error
        )

    def test_every_site_calls_the_one_refusal(self) -> None:
        """One refusal, one text: the fit layer, the evaluators and the replay share one check.

        ``cleverly.msm`` defines the check that the model and ``MSMSet.evaluate`` call.
        """
        assert tmle_module.refuse_msm_functions is refuse_msm_functions
        assert ltmle_module.refuse_msm_functions is refuse_msm_functions
        assert regimen_msm_module.refuse_msm_functions is refuse_msm_functions
        assert replay_module._design_kind is msm_module._design_kind


# ------------------------------------------------------------------ the witness

#: The law of the witness.  ``Qbar`` rises in ``W`` in both arms, so the ``W`` coefficient
#: is material, and every cell is a multiple of ``1 / N``, so the 1000 rows realise the law
#: exactly.  ``P(W)`` is the default, whose mean of ``W`` is 0.7.
CENTRE_COUNTS = law.cell_counts(q=[[0.1, 0.2], [0.5, 0.6], [0.8, 0.9]])
CENTRE_PROBS = CENTRE_COUNTS / law.N
CELL_P = cell_p(CENTRE_PROBS)
W_OF = np.array([w for w, _, _ in law.SUPPORT], dtype=float)
#: Uniform projection weights, as the ``(3, 2)`` array of the oracle.
UNIFORM = np.ones((3, 2))
#: The ``W`` column of the oracle's ``(3, 2, 3)`` design, the one the centre moves.
W_COLUMN = np.array([0.0, 0.0, 1.0])
#: The sample mean of ``W`` on the exact sample, and so the centre the user froze.
CENTRE = 0.7


def centred(probs: Any) -> Any:
    """``phi = [1, a, W - E_P[W]]``: a design whose centre moves with the law."""
    p_w = np.asarray(probs).sum(axis=(1, 2))
    return law.MSM_DESIGN - (p_w @ np.arange(3.0)) * W_COLUMN


#: ``phi = [1, a, W - 0.7]``: the design with the centre frozen at its value on this law.
FIXED_DESIGN = law.MSM_DESIGN - CENTRE * W_COLUMN


def design_eif(design: Any) -> np.ndarray:
    """``(12, 3)``: the Gateaux derivative of the uniform-weight projection of ``design``."""
    return msm_eif(CENTRE_PROBS, UNIFORM, design)


def sample_centre(w: Any) -> CentredDesign:
    """The user lie: ``W`` centred at its sample mean, a statistic of the sample."""
    return CentredDesign(float(np.mean(w)))


def centre_oracle_fit(model: MSM) -> Any:
    """``model`` fitted on the witness law with its own nuisances."""
    return oracle_fit(CENTRE_COUNTS, msm=model, estimands="all")


@pytest.fixture(scope="module")
def centre_fit() -> Any:
    """The sample centre, declared ``"known"``: the lie that no declaration can detect."""
    sample = law.frame(CENTRE_COUNTS)
    return centre_oracle_fit(written(sample_centre(sample["W"]), design_kind="known"))


#: The reported SE over the exact estimated-centre SE for the intercept must stay below
#: this.  The plan measured 0.8927 before it chose the bound (RM27 in docs/roadmap.md).  A
#: reported curve that carried the centre term would give 1.  The bound claims an
#: understatement of more than 5 percent and pins no digit; the approx line records the
#: measured value.
UNDERSTATEMENT_BOUND = 0.95

INTERCEPT = law.MSM_TERMS.index("(intercept)")
W_TERM = law.MSM_TERMS.index("W")


def centre_term() -> np.ndarray:
    """``T = beta_W (W - 0.7)``, the intercept term the fixed-centre curve omits."""
    beta = law.msm_beta(CENTRE_PROBS, UNIFORM, design=FIXED_DESIGN)
    return beta[W_TERM] * (W_OF - CENTRE)


class TestAnEstimatedCentreMisstatesTheVariance:
    def test_the_sample_centre_is_the_fixed_centre(self, centre_fit: Any) -> None:
        """The premise: the sample mean is 0.7, and the fit is the fixed-centre projection."""
        assert sample_centre(law.frame(CENTRE_COUNTS)["W"]).centre == pytest.approx(
            CENTRE, abs=1e-12
        )
        beta = law.msm_beta(CENTRE_PROBS, UNIFORM, design=FIXED_DESIGN)
        for index, term in enumerate(law.MSM_TERMS):
            assert centre_fit.estimates[f"msm[{term}]"].psi == pytest.approx(beta[index], abs=1e-12)

    @pytest.mark.parametrize("index", range(len(law.MSM_TERMS)))
    def test_control_the_curve_is_the_fixed_centre_eif(self, centre_fit: Any, index: int) -> None:
        """Control: the curve is right for the functional the user declared."""
        np.testing.assert_allclose(
            msm_curve(centre_fit, index)[law.first_row_of(CENTRE_COUNTS)],
            design_eif(FIXED_DESIGN)[:, index],
            atol=1e-10,
            rtol=0,
        )

    def test_the_oracle_carries_exactly_the_centre_term(self) -> None:
        """The two oracles differ by ``T`` in the intercept, and by nothing in ``a`` or ``W``."""
        gap = design_eif(centred) - design_eif(FIXED_DESIGN)
        term = centre_term()
        np.testing.assert_allclose(gap[:, INTERCEPT], term, atol=1e-12, rtol=0)
        np.testing.assert_allclose(np.delete(gap, INTERCEPT, axis=1), 0.0, atol=1e-12, rtol=0)
        # The nonzero witness: the measured E[T^2] is 0.0779.
        assert float(CELL_P @ term**2) > 0.05

    def test_witness_the_intercept_understates_its_standard_error(self, centre_fit: Any) -> None:
        ratio = se_ratio(
            msm_curve(centre_fit, INTERCEPT), design_eif(centred)[:, INTERCEPT], CELL_P
        )
        assert ratio == pytest.approx(0.8927, abs=1e-4)
        assert ratio < UNDERSTATEMENT_BOUND

    @pytest.mark.parametrize("term", ["a", "W"])
    def test_control_the_other_coefficients_are_unaffected(
        self, centre_fit: Any, term: str
    ) -> None:
        """Control: the centre shifts the intercept only, so ``a`` and ``W`` read 1."""
        index = law.MSM_TERMS.index(term)
        ratio = se_ratio(msm_curve(centre_fit, index), design_eif(centred)[:, index], CELL_P)
        assert abs(ratio - 1.0) < 1e-9

    def test_mutation_a_frozen_oracle_loses_the_witness(self, centre_fit: Any) -> None:
        """Mutation M6: an oracle that froze the centre would read 1 and fail the bound."""
        frozen = se_ratio(
            msm_curve(centre_fit, INTERCEPT), design_eif(FIXED_DESIGN)[:, INTERCEPT], CELL_P
        )
        assert frozen == pytest.approx(1.0, abs=1e-9)
        assert not frozen < UNDERSTATEMENT_BOUND

    def test_an_estimated_declaration_is_refused_on_this_law(self) -> None:
        """The honest declaration of the sample centre meets the refusal, before any fit."""
        centre = sample_centre(law.frame(CENTRE_COUNTS)["W"])
        assert_refused(lambda: written(centre, design_kind="estimated"), CapabilityError, ESTIMATED)
        model = restored(written(centre, design_kind="known"), "design_kind", "estimated")
        assert_refused(lambda: tmle_fit(model, never_fit_learners()), CapabilityError, ESTIMATED)
        assert NeverFit.calls == 0


class TestAKnownDesignKeepsItsInterval:
    def test_a_fixed_centre_reports_the_fixed_centre_eif(self, centre_fit: Any) -> None:
        """A stated centre of 0.7 reports the curve of the sample centre, bit for bit.

        The control above checks that curve against the fixed-centre Gateaux curve.
        """
        result = centre_oracle_fit(written(CentredDesign(CENTRE), design_kind="known"))
        for index, term in enumerate(law.MSM_TERMS):
            estimate = result.estimates[f"msm[{term}]"]
            assert estimate.inference == "influence_curve"
            assert np.all(np.isfinite(estimate.ci))
            np.testing.assert_array_equal(msm_curve(result, index), msm_curve(centre_fit, index))

    def test_the_shorthand_reports_its_eif(self) -> None:
        result = centre_oracle_fit(linear())
        rows = law.first_row_of(CENTRE_COUNTS)
        for index, term in enumerate(law.MSM_TERMS):
            estimate = result.estimates[f"msm[{term}]"]
            assert estimate.inference == "influence_curve"
            assert np.all(np.isfinite(estimate.ci))
            np.testing.assert_allclose(
                msm_curve(result, index)[rows], design_eif(None)[:, index], atol=1e-10, rtol=0
            )
