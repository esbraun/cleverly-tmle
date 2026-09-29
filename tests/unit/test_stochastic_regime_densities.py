r"""A ``Stochastic`` regime density must be declared known, and an estimated one is refused.

A regime density :math:`g^\star(a \mid W)` is part of the estimand.  When it is a fixed
function of the covariates, the influence curve that the package reports for a regime is the
efficient influence function of the regime mean.  When the target is the population-law
odds tilt of the mechanism, :math:`g^\star` is a functional of :math:`P`.  Its efficient
influence function then carries a further pathwise-derivative term that the regime curve
lacks.  A realized learned density defines a different, data-adaptive target.

A callable can close over any estimate, and no code can inspect a closure, so the status of
the density is a declaration: ``Stochastic(density_kind=...)``.  RM25 in ``docs/roadmap.md``
records the defect.  This module pins these things:

* the declaration is required, ``"estimated"`` is refused, and a density that is not
  callable is refused, with their messages;
* the positional order of ``Stochastic`` is unchanged;
* every fit entry refuses a modified regime before any learner or density call, and so do
  ``RegimeSet.evaluate`` and ``Stochastic.density`` called directly;
* every recomputation from a result whose regime loses its declaration refuses;
* the simulated-confounding replay admits a frozen regime and refuses an undeclared one
  before its density runs;
* a deliberate mutation that removes a refusal makes those witnesses fail;
* on an exact law, the odds tilt of the sample mechanism declared ``"known"`` gets the
  fixed-density curve, which understates variance for the population-law odds-tilt target;
* a density that is known keeps its interval.

The exact-law gap is the reason for the refusal, measured without sampling error, as
``tests/unit/test_msm_projection_weights.py`` measures it for RM13.  The law and its closed
forms are in ``tests/unit/_tilt_law_support.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import pytest

import cleverly._declarations as declarations_module
import cleverly.interventions.base as base_module
from cleverly import RegimeMean
from cleverly.assessment import replayability
from cleverly.data import CausalData
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError, DataError
from cleverly.interventions import Incremental, RegimeSet, Static, Stochastic
from cleverly.interventions.base import (
    _ESTIMATED_DENSITY,
    _ESTIMATED_INTERVENTION,
    _ESTIMATED_RULE,
    _UNDECLARED_DENSITY,
    _UNDECLARED_INTERVENTION,
    _UNDECLARED_RULE,
    refuse_regime_densities,
)
from cleverly.msm import (
    _ESTIMATED_DESIGN,
    _ESTIMATED_WEIGHTS,
    _UNDECLARED_DESIGN,
    _UNDECLARED_WEIGHTS,
)
from cleverly.sensitivity import _simulated_confounding_fixed as replay_module
from cleverly.sensitivity import simulated_confounding
from tests import discrete_law as law
from tests.conftest import linear_in_sample
from tests.unit._confounding_support import Counter, forbid_draw_and_refit, validate_replay
from tests.unit._declaration_support import (
    PATHWISE,
    assert_every_witness_fails,
    assert_refused,
    assert_refused_before_any_call,
    assert_replay_agrees,
    modified,
    modified_states,
    oracle_fit,
    point_entries,
    recomputations,
    se_ratio,
    tmle_module,
    undeclared_copy,
)
from tests.unit._msm_declaration_support import (
    DESIGN_UNKNOWN,
    WEIGHTS_UNKNOWN,
    FixedWeight,
    linear,
    written,
)
from tests.unit._natural_course_support import NeverFit, never_fit_learners
from tests.unit._policy_declaration_support import (
    INTERVENTION_UNKNOWN,
    RULE_UNKNOWN,
    DataTilt,
    checked,
    threshold_regimen,
    threshold_rule,
)
from tests.unit._simulated_confounding_support import (
    _GRID,
    _alias,
    _estimate,
    _fit_policy,
    _stochastic_density,
    _study,
)
from tests.unit._tilt_law_support import (
    CELL_P,
    DELTAS,
    TILT_COUNTS,
    TILT_PROBS,
    UNDERSTATEMENT_BOUND,
    KnownTilt,
    SampleTilt,
    fixed_eif,
    kennedy_term,
    tilt_curve,
)

#: The two declaration refusals, imported from the module that raises them, so the text
#: is written once.
UNDECLARED = _UNDECLARED_DENSITY
ESTIMATED = _ESTIMATED_DENSITY
#: Fragments of the other refusals.  A test matches a fragment, not the whole text, so a
#: rewording of the explanation does not break it, but a message from another check does.
NOT_CALLABLE = "density_fn= must be callable"
UNKNOWN = "density_kind must be 'known', 'estimated' or None"


@dataclass(frozen=True)
class FixedDensity:
    """A known density, 0.4 and 0.6 in every row.  A class rather than a lambda, so it pickles."""

    def __call__(self, frame: Any) -> np.ndarray:
        return np.column_stack([np.full(len(frame), 0.4), np.full(len(frame), 0.6)])


def coin(density: Any = None, **declaration: Any) -> Stochastic:
    return Stochastic(FixedDensity() if density is None else density, "coin", **declaration)


def causal_data() -> CausalData:
    return CausalData.from_frame(law.frame(), outcome="Y", treatment="A")


# ------------------------------------------------------------------ the declaration


class TestTheDeclarationIsRequired:
    def test_an_undeclared_density_is_refused(self) -> None:
        assert_refused(coin, CapabilityError, UNDECLARED, "density_kind='known'")

    def test_an_estimated_density_is_refused_by_its_missing_term(self) -> None:
        assert_refused(
            lambda: coin(density_kind="estimated"),
            CapabilityError,
            ESTIMATED,
            PATHWISE,
            "population-law target",
            "data-adaptive target",
            "TMLE(incremental=",
            "IncrementalMean and IncrementalEffect",
        )

    @pytest.mark.parametrize("kind", ["Known", "probability", True, 1])
    def test_an_unknown_declaration_is_refused(self, kind: Any) -> None:
        assert_refused(lambda: coin(density_kind=kind), DataError, UNKNOWN)

    @pytest.mark.parametrize("density", [0.6, np.ones((10, 2))], ids=["float", "array"])
    @pytest.mark.parametrize("kind", [None, "known", "estimated"])
    def test_a_density_that_is_not_callable_is_refused(self, density: Any, kind: Any) -> None:
        """The callable check runs first, so every declaration meets the same refusal."""
        assert_refused(lambda: coin(density, density_kind=kind), DataError, NOT_CALLABLE)

    def test_a_known_density_is_accepted_and_evaluates(self) -> None:
        regime = coin(density_kind="known")
        assert regime.density_kind == "known"
        data = causal_data()
        np.testing.assert_array_equal(regime.density(data), FixedDensity()(np.zeros(data.n)))

    def test_a_subclass_inherits_the_declaration(self) -> None:
        class Declared(Stochastic):
            pass

        assert_refused(lambda: Declared(FixedDensity(), "coin"), CapabilityError, UNDECLARED)
        assert Declared(FixedDensity(), "coin", density_kind="known").density_kind == "known"


class TestThePositionalOrderIsUnchanged:
    """``density_kind`` is the last field, so ``Stochastic(density_fn, name)`` keeps its order."""

    def test_the_declaration_is_the_last_field(self) -> None:
        assert [field.name for field in fields(Stochastic)] == [
            "density_fn",
            "name",
            "density_kind",
        ]

    def test_the_third_positional_argument_is_the_declaration(self) -> None:
        assert Stochastic(FixedDensity(), "coin", "known").density_kind == "known"
        assert_refused(lambda: Stochastic(FixedDensity(), "coin"), CapabilityError, UNDECLARED)


# ------------------------------------------------------------------ the fit layer


def spy_coin() -> Stochastic:
    """A known regime whose density counts its calls in ``regime.density_fn.calls``."""
    return coin(Counter(FixedDensity()), density_kind="known")


#: Every fit entry that can reach a modified regime: ``TMLE.fit``, ``CausalStudy.estimate``,
#: and ``TMLE.refit``.
ENTRIES = point_entries(
    lambda regime: {"interventions": (regime,)}, lambda regime: RegimeMean((regime,))
)

#: What a modified regime can carry, and the refusal each one meets.
MODIFIED = modified_states(UNDECLARED, ESTIMATED)


def assert_entry_refuses(entry: str, regime: Any, *fragments: str) -> None:
    assert_refused_before_any_call(
        lambda: ENTRIES[entry](regime, never_fit_learners()),
        regime.density_fn,
        "density",
        *fragments,
    )


class TestTheFitRefusesAModifiedRegime:
    @pytest.mark.parametrize("entry", list(ENTRIES))
    @pytest.mark.parametrize("name", list(MODIFIED))
    def test_every_entry_refuses_before_any_learner_or_density_call(
        self, entry: str, name: str
    ) -> None:
        kind, fragments = MODIFIED[name]
        assert_entry_refuses(entry, modified(spy_coin(), "density_kind", kind), *fragments)

    def test_a_subclass_that_skips_the_declaration_refuses_at_the_fit(self) -> None:
        """The fit selects regimes with ``isinstance``, so a subclass cannot opt out."""

        class Skips(Stochastic):
            def __post_init__(self) -> None:
                pass

        regime = Skips(Counter(FixedDensity()), "coin")
        assert regime.density_kind is None
        assert_entry_refuses("fit", regime, UNDECLARED)

    @pytest.mark.parametrize("entry", list(ENTRIES))
    def test_a_known_declaration_reaches_the_first_learner(self, entry: str) -> None:
        """The control: the same entries with the declaration intact go past the check."""
        regime = spy_coin()
        with pytest.raises(AssertionError, match="before any learner is fitted"):
            ENTRIES[entry](regime, never_fit_learners())
        assert NeverFit.calls == 1
        assert regime.density_fn.calls == 0, "a density ran before the first learner"


#: Each public evaluator that runs a density, called directly as a user can call it.
EVALUATORS: dict[str, Callable[[Stochastic], Any]] = {
    "RegimeSet.evaluate": lambda regime: RegimeSet.evaluate((regime,), causal_data()),
    "Stochastic.density": lambda regime: regime.density(causal_data()),
}


class TestTheEvaluatorsCheckFirst:
    """A modified regime handed straight to an evaluator refuses before its density runs."""

    @pytest.mark.parametrize("evaluator", list(EVALUATORS))
    @pytest.mark.parametrize("name", list(MODIFIED))
    def test_a_modified_regime_refuses_before_its_density_runs(
        self, evaluator: str, name: str
    ) -> None:
        kind, fragments = MODIFIED[name]
        regime = modified(spy_coin(), "density_kind", kind)
        assert_refused(lambda: EVALUATORS[evaluator](regime), CapabilityError, *fragments)
        assert regime.density_fn.calls == 0, "the density was evaluated before the refusal"

    def test_the_set_checks_every_regime_before_the_first_density(self) -> None:
        """A declared regime listed first does not run before a modified one refuses."""
        first = Stochastic(Counter(FixedDensity()), "first", density_kind="known")
        regime = modified(spy_coin(), "density_kind", None)
        assert_refused(
            lambda: RegimeSet.evaluate((first, regime), causal_data()), CapabilityError, UNDECLARED
        )
        assert first.density_fn.calls == 0, "the first density ran before the refusal"
        assert regime.density_fn.calls == 0

    @pytest.mark.parametrize("evaluator", list(EVALUATORS))
    def test_a_declared_regime_is_evaluated(self, evaluator: str) -> None:
        """The control: the same call with the declaration intact runs the density."""
        regime = spy_coin()
        EVALUATORS[evaluator](regime)
        assert regime.density_fn.calls == 1


# ------------------------------------------------------------------ a lost declaration


def undeclared_regime(regime: Stochastic) -> Stochastic:
    """``regime`` with its declaration removed after construction."""
    return modified(regime, "density_kind", None)


class TestARegimeThatLosesItsDeclaration:
    def test_it_cannot_be_replaced_without_a_declaration(self) -> None:
        old = undeclared_regime(coin(density_kind="known"))
        assert old.density_kind is None
        assert_refused(lambda: replace(old), CapabilityError, UNDECLARED)
        assert replace(old, density_kind="known").density_kind == "known"

    def test_it_refuses_at_the_fit(self) -> None:
        assert_entry_refuses("fit", undeclared_regime(spy_coin()), UNDECLARED)


def stochastic_regimes(result: Any) -> list[Any]:
    return [item for item in result.estimator.interventions if isinstance(item, Stochastic)]


def undeclared(result: Any) -> Any:
    """A copy of ``result`` whose stochastic regimes no longer declare their densities."""
    return undeclared_copy(result, "density_kind", stochastic_regimes)


#: The estimands a retarget of the undeclared result requests.
RETARGETED = ("ey_regime", "ate_regime")


class TestAnUndeclaredResultRefusesARecomputation:
    """RM25: a result whose density loses its declaration computes nothing new.

    Every sweep recomputes through ``_retarget_detailed``, and a refit through
    ``_resolve_estimands_for_data``, and both check the declaration as the fit does.
    """

    @pytest.fixture(scope="class")
    def result(self) -> Any:
        regimes = (Static(0, name="never"), coin(density_kind="known"))
        estimator = TMLE(interventions=regimes, **linear_in_sample())
        return estimator.fit(law.frame(), outcome="Y", treatment="A").single()

    def test_the_replay_slots_read_false(self, result: Any) -> None:
        assert "ey_regime[coin]" in result.estimates
        old = undeclared(result)
        assert not replayability(old).retarget_cached_nuisances
        assert not replayability(old).refit_nuisances
        assert not old.diagnostics.capability("truncation_curve").available
        assert replayability(result).refit_nuisances

    def test_each_replay_slot_agrees_with_its_call(self, result: Any) -> None:
        """Both slots of the undeclared result read false, and both calls refuse."""
        assert_replay_agrees(undeclared(result), RETARGETED)
        assert_replay_agrees(result, RETARGETED)

    @pytest.mark.parametrize("entry", ["truncation_curve", "retarget", "refit"])
    def test_every_recomputation_refuses(self, result: Any, entry: str) -> None:
        old = undeclared(result)
        assert_refused(recomputations(old, RETARGETED)[entry], CapabilityError, UNDECLARED)

    @pytest.mark.parametrize("entry", ["truncation_curve", "retarget", "refit"])
    def test_the_declared_result_recomputes(self, result: Any, entry: str) -> None:
        """The control: the same entry on the result before the declaration was lost."""
        recomputations(result, RETARGETED)[entry]()

    @pytest.mark.parametrize("entry", ["truncation_curve", "retarget", "refit"])
    def test_removing_the_fit_layer_check_fails_the_refusal(
        self, result: Any, entry: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A refit evaluates the densities again, so its mutation removes those checks too."""
        old = undeclared(result)
        monkeypatch.setattr(tmle_module, "refuse_regime_densities", lambda interventions: None)
        if entry == "refit":
            monkeypatch.setattr(base_module, "refuse_regime_densities", lambda items: None)
        with pytest.raises(AssertionError):
            assert_refused(recomputations(old, RETARGETED)[entry], CapabilityError, UNDECLARED)


# ------------------------------------------------------------------ the replay


def counted_fit() -> tuple[Any, Counter]:
    """A fitted regime mean whose density counts its calls, without its declaration.

    The counter is read off the copied result, because a pickle round trip copies it.
    """
    regime = Stochastic(Counter(_stochastic_density), name="policy", density_kind="known")
    old = undeclared(_estimate(_study(), RegimeMean((regime,))))
    density = old.estimator.interventions[0].density_fn
    density.calls = 0
    return old, density


class TestTheReplay:
    def test_a_known_regime_replays_through_frozen_regimes(self) -> None:
        result = _fit_policy()
        replay = validate_replay(result)
        assert replay.interventions
        assert all(type(item) is replay_module._FrozenRegime for item in replay.interventions)
        assert refuse_regime_densities(replay.interventions) is None
        surface = simulated_confounding(
            result, estimand=_alias(result), grid=_GRID, random_state=31
        )
        assert all(cell.failure is None for cell in surface.cells)

    def test_an_undeclared_regime_refuses_at_replay_before_its_density_runs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        old, density = counted_fit()
        forbid_draw_and_refit(monkeypatch, old.estimator)
        assert_refused(
            lambda: simulated_confounding(old, estimand=_alias(old), grid=_GRID, random_state=31),
            CapabilityError,
            UNDECLARED,
        )
        assert_refused(lambda: validate_replay(old), CapabilityError, UNDECLARED)
        assert density.calls == 0, "the density was evaluated before the refusal"

    def test_removing_the_evaluator_check_evaluates_the_density(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The deliberate-mutation control: without the check the replay runs the density.

        ``RegimeSet.evaluate`` and ``Stochastic.density`` read the module global, so the
        mutation removes both.  Each frozen regime carries the source's ``None`` (roadmap row
        RM28), and ``TMLE`` holds its own reference to the check, so the refit still refuses.
        The refusal arrives after the density ran, which is what the check removes.
        """
        old, density = counted_fit()
        monkeypatch.setattr(base_module, "refuse_regime_densities", lambda interventions: None)
        replay = validate_replay(old)
        assert all(type(item) is replay_module._FrozenRegime for item in replay.interventions)
        assert density.calls == 1
        assert_refused(lambda: replay.refit(old.data), CapabilityError, _UNDECLARED_INTERVENTION)


# ------------------------------------------------------------------ mutation controls


def declaration_witnesses() -> list[Callable[[], None]]:
    """Every declaration-layer witness above, as a call that must raise to pass."""
    suite = TestTheDeclarationIsRequired()
    return [
        suite.test_an_undeclared_density_is_refused,
        suite.test_an_estimated_density_is_refused_by_its_missing_term,
        lambda: suite.test_an_unknown_declaration_is_refused("Known"),
        lambda: suite.test_a_density_that_is_not_callable_is_refused(0.6, "known"),
    ]


class TestTheWitnessesHaveTeeth:
    def test_removing_the_declaration_check_fails_every_declaration_witness(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(base_module, "refuse_regime_densities", lambda interventions: None)
        assert_every_witness_fails(declaration_witnesses())

    @pytest.mark.parametrize("entry", list(ENTRIES))
    @pytest.mark.parametrize("name", list(MODIFIED))
    def test_removing_the_fit_layer_check_fails_the_fit_witnesses(
        self, entry: str, name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        kind, fragments = MODIFIED[name]
        monkeypatch.setattr(tmle_module, "refuse_regime_densities", lambda interventions: None)
        with pytest.raises(AssertionError):
            assert_entry_refuses(entry, modified(spy_coin(), "density_kind", kind), *fragments)
        assert NeverFit.calls > 0, "the mutated fit refused before a learner"

    def test_every_site_calls_the_one_refusal(self) -> None:
        """One refusal, one text: the fit layer calls the check that the evaluators call.

        ``cleverly.interventions.base`` defines that check, and its evaluators call it.
        """
        assert tmle_module.refuse_regime_densities is refuse_regime_densities

    def test_removing_the_shared_declaration_fails_each_of_its_users(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RM13, RM25, RM27 and RM28 refuse through one ``FunctionDeclaration.refuse``.

        Each witness that the shared refusal decides refuses before the mutation and fails
        under it.
        """
        witnesses = [
            partial(assert_refused, build, error, *fragments)
            for build, error, fragments in SHARED_REFUSALS.values()
        ]
        for witness in witnesses:
            witness()
        monkeypatch.setattr(
            declarations_module.FunctionDeclaration, "refuse", lambda self, kind: None
        )
        assert_every_witness_fails(witnesses)
        assert linear(weights=FixedWeight()).weights_kind is None
        assert coin().density_kind is None
        assert written().design_kind is None
        assert threshold_rule().rule_kind is None
        assert threshold_regimen().rule_kind is None


#: Each refusal that ``FunctionDeclaration.refuse`` decides: a build, the error it raises,
#: and the fragments of its message.  The RM13 unknown-value refusal is not among them:
#: ``refuse_msm_functions`` calls ``check`` itself for the weight, so it still refuses.
SHARED_REFUSALS: dict[str, tuple[Callable[[], Any], type[Exception], tuple[str, ...]]] = {
    "undeclared weight": (
        lambda: linear(weights=FixedWeight()),
        CapabilityError,
        (_UNDECLARED_WEIGHTS, "'known'"),
    ),
    "estimated weight": (
        lambda: linear(weights=FixedWeight(), weights_kind="estimated"),
        CapabilityError,
        (_ESTIMATED_WEIGHTS, PATHWISE),
    ),
    "undeclared density": (coin, CapabilityError, (UNDECLARED, "density_kind='known'")),
    "estimated density": (
        lambda: coin(density_kind="estimated"),
        CapabilityError,
        (ESTIMATED, PATHWISE),
    ),
    "unknown density": (lambda: coin(density_kind="Known"), DataError, (UNKNOWN,)),
    "undeclared design": (written, CapabilityError, (_UNDECLARED_DESIGN, "design_kind='known'")),
    "estimated design": (
        lambda: written(design_kind="estimated"),
        CapabilityError,
        (_ESTIMATED_DESIGN, PATHWISE),
    ),
    "unknown design": (lambda: written(design_kind="Known"), DataError, (DESIGN_UNKNOWN,)),
    "undeclared rule": (threshold_rule, CapabilityError, (_UNDECLARED_RULE, "rule_kind='known'")),
    "estimated rule": (
        lambda: threshold_rule(rule_kind="estimated"),
        CapabilityError,
        (_ESTIMATED_RULE, PATHWISE),
    ),
    "unknown rule": (lambda: threshold_rule(rule_kind="Known"), DataError, (RULE_UNKNOWN,)),
    "undeclared regimen": (
        threshold_regimen,
        CapabilityError,
        (_UNDECLARED_RULE, "DynamicRegimen(label, plan, rule_kind='known')"),
    ),
    "estimated regimen": (
        lambda: threshold_regimen(rule_kind="estimated"),
        CapabilityError,
        (_ESTIMATED_RULE, PATHWISE),
    ),
    "unknown regimen": (
        lambda: threshold_regimen(rule_kind="Known"),
        DataError,
        (RULE_UNKNOWN,),
    ),
    "undeclared intervention": (
        lambda: checked(DataTilt()),
        CapabilityError,
        (_UNDECLARED_INTERVENTION, "density_kind attribute of 'known'"),
    ),
    "estimated intervention": (
        lambda: checked(DataTilt(density_kind="estimated")),
        CapabilityError,
        (_ESTIMATED_INTERVENTION, PATHWISE),
    ),
}


# ------------------------------------------------------------------ the shared check

#: Each user of the shared declaration: a build from one declaration value, the fragment of
#: its unknown-value refusal, and its field.  The uniform MSM has no weight, so its own check
#: of ``"estimated"`` without a weight must not read a value that is not a string first.
DECLARATION_USERS: dict[str, tuple[Callable[[Any], Any], str, str]] = {
    "msm weight": (
        lambda kind: linear(weights=FixedWeight(), weights_kind=kind),
        WEIGHTS_UNKNOWN,
        "weights_kind",
    ),
    "uniform msm": (lambda kind: linear(weights_kind=kind), WEIGHTS_UNKNOWN, "weights_kind"),
    "regime density": (lambda kind: coin(density_kind=kind), UNKNOWN, "density_kind"),
    "msm design": (lambda kind: written(design_kind=kind), DESIGN_UNKNOWN, "design_kind"),
    "rule": (lambda kind: threshold_rule(rule_kind=kind), RULE_UNKNOWN, "rule_kind"),
    "regimen": (lambda kind: threshold_regimen(rule_kind=kind), RULE_UNKNOWN, "rule_kind"),
    "intervention": (
        lambda kind: checked(DataTilt(density_kind=kind)),
        INTERVENTION_UNKNOWN,
        "density_kind",
    ),
}

#: Values that are not a string.  Before the check tested the type, the first built, the
#: second raised the ``ValueError`` of an array's truth value, and the third raised the
#: ``TypeError`` of ``pandas.NA``.
NOT_A_STRING = {
    "array['known']": np.array(["known"]),
    "array['known', 'known']": np.array(["known", "known"]),
    "pandas.NA": pd.NA,
}


class TestTheSharedCheckRefusesAValueThatIsNotAString:
    @pytest.mark.parametrize("user", list(DECLARATION_USERS))
    @pytest.mark.parametrize("value", list(NOT_A_STRING))
    def test_a_value_that_is_not_a_string_is_an_unknown_declaration(
        self, user: str, value: str
    ) -> None:
        build, fragment, _ = DECLARATION_USERS[user]
        assert_refused(lambda: build(NOT_A_STRING[value]), DataError, fragment)

    @pytest.mark.parametrize("user", list(DECLARATION_USERS))
    def test_a_numpy_string_is_a_string(self, user: str) -> None:
        """The control: ``numpy.str_`` is a ``str`` subclass, so ``"known"`` in it builds."""
        build, _, field = DECLARATION_USERS[user]
        assert getattr(build(np.str_("known")), field) == "known"


# ------------------------------------------------------------------ the witness


@pytest.fixture(scope="module")
def sample_fits() -> dict[str, Any]:
    """The sample tilt, declared ``"known"``: the lie that no declaration can detect."""
    sample = law.frame(TILT_COUNTS)
    return {
        label: oracle_fit(
            TILT_COUNTS,
            interventions=(Stochastic(SampleTilt(sample, delta), "tilt", density_kind="known"),),
        )
        for label, delta in DELTAS.items()
    }


class TestAnEstimatedTiltUnderstatesTheVariance:
    def test_the_sample_tilt_is_the_known_tilt(self) -> None:
        """The premise: on the exact sample the two densities agree."""
        for delta in DELTAS.values():
            np.testing.assert_allclose(
                SampleTilt(law.frame(TILT_COUNTS), delta).star,
                KnownTilt(delta).star,
                atol=1e-15,
                rtol=0,
            )

    @pytest.mark.parametrize("label", list(DELTAS))
    def test_control_the_curve_is_the_fixed_density_eif(self, sample_fits, label: str) -> None:
        """Control: the curve is right for the functional the user declared."""
        np.testing.assert_allclose(
            tilt_curve(sample_fits[label])[law.first_row_of(TILT_COUNTS)],
            fixed_eif(KnownTilt(DELTAS[label]).star),
            atol=1e-12,
            rtol=0,
        )

    @pytest.mark.parametrize("label", list(DELTAS))
    def test_the_oracle_carries_exactly_the_kennedy_term(self, label: str) -> None:
        delta = DELTAS[label]
        gap = law.eif(f"ey_ipsi[{label}]", probs=TILT_PROBS) - fixed_eif(KnownTilt(delta).star)
        np.testing.assert_allclose(gap, kennedy_term(delta), atol=1e-12, rtol=0)
        # The nonzero witness: the measured E[T^2] is 0.1603 at delta = 2 and 0.1158 at 0.5.
        assert float(CELL_P @ kennedy_term(delta) ** 2) > 0.1

    @pytest.mark.parametrize("label", list(DELTAS))
    def test_the_omitted_term_is_orthogonal_to_the_reported_curve(self, label: str) -> None:
        """So the omission understates the variance by exactly ``Var(T)`` at the truth.

        The probe measured the cross moment at 8.7e-19 and 3.5e-18, and the variance gap
        at 0 and 1.4e-17.
        """
        delta = DELTAS[label]
        fixed, term = fixed_eif(KnownTilt(delta).star), kennedy_term(delta)
        exact = law.eif(f"ey_ipsi[{label}]", probs=TILT_PROBS)
        assert float(CELL_P @ (fixed * term)) == pytest.approx(0.0, abs=1e-12)
        assert float(CELL_P @ exact**2) == pytest.approx(
            float(CELL_P @ fixed**2) + float(CELL_P @ term**2), abs=1e-12
        )

    def test_witness_the_reported_se_understates_the_exact_se(self, sample_fits) -> None:
        ratio = se_ratio(
            tilt_curve(sample_fits["odds x2"]),
            law.eif("ey_ipsi[odds x2]", probs=TILT_PROBS),
            CELL_P,
        )
        assert ratio == pytest.approx(0.6226, abs=1e-4)
        assert ratio < UNDERSTATEMENT_BOUND

    def test_the_other_side_of_one_is_recorded(self, sample_fits) -> None:
        """delta = 0.5 tilts toward control; the ratio is recorded, and no bound is claimed."""
        ratio = se_ratio(
            tilt_curve(sample_fits["odds x0.5"]),
            law.eif("ey_ipsi[odds x0.5]", probs=TILT_PROBS),
            CELL_P,
        )
        assert ratio == pytest.approx(0.6853, abs=1e-4)

    def test_mutation_a_frozen_oracle_loses_the_witness(self, sample_fits) -> None:
        """An oracle that froze ``g`` would read 1 and fail the bound."""
        frozen = se_ratio(
            tilt_curve(sample_fits["odds x2"]), fixed_eif(KnownTilt(2.0).star), CELL_P
        )
        assert frozen == pytest.approx(1.0, abs=1e-12)
        assert not frozen < UNDERSTATEMENT_BOUND

    def test_the_package_ipsi_curve_differs_by_the_same_term(self, sample_fits) -> None:
        """The incremental axis carries the term that the regime curve omits."""
        ipsi = oracle_fit(TILT_COUNTS, incremental=(Incremental(2.0, name="odds x2"),))
        rows = law.first_row_of(TILT_COUNTS)
        gap = (
            np.asarray(ipsi.estimates["ey_ipsi[odds x2]"].influence_curve)[rows]
            - tilt_curve(sample_fits["odds x2"])[rows]
        )
        np.testing.assert_allclose(gap, kennedy_term(2.0), atol=1e-12, rtol=0)

    def test_an_estimated_declaration_is_refused_on_this_law(self) -> None:
        """The honest declaration of the sample tilt meets the refusal, before any fit."""
        tilt = SampleTilt(law.frame(TILT_COUNTS), 2.0)
        assert_refused(
            lambda: Stochastic(tilt, "tilt", density_kind="estimated"),
            CapabilityError,
            ESTIMATED,
            "TMLE(incremental=",
        )


class TestAKnownDensityKeepsItsInterval:
    def test_a_known_tilt_reports_the_fixed_density_eif_under_influence_curve(self) -> None:
        result = oracle_fit(
            TILT_COUNTS, interventions=(Stochastic(KnownTilt(2.0), "tilt", density_kind="known"),)
        )
        estimate = result.estimates["ey_regime[tilt]"]
        assert estimate.inference == "influence_curve"
        assert np.all(np.isfinite(estimate.ci))
        np.testing.assert_allclose(
            tilt_curve(result)[law.first_row_of(TILT_COUNTS)],
            fixed_eif(KnownTilt(2.0).star),
            atol=1e-12,
            rtol=0,
        )
