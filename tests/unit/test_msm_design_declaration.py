r"""An MSM design must be declared known, and an estimated one is refused.

The working design :math:`\varphi(a, V)` is part of the estimand.  When it is a fixed
function, the influence curve the package reports is the efficient influence function of
:math:`\beta`.  When it is computed from the sample, such as a covariate centred at its sample
mean, :math:`\varphi` is a functional of :math:`P`.  The efficient influence function then
carries a further term for the pathwise derivative through that statistic, and the reported
curve does not have it.

A callable can close over any estimate, and no code can inspect a closure, so the status of
the design is a declaration: ``MSM(design_kind=...)``.  ``MSM.linear`` declares its own
design known.  RM27 in ``docs/roadmap.md`` records the defect.  This module pins these
things:

* the declaration is required, ``"estimated"`` is refused, and a design that is not callable
  is refused, with their messages;
* a model pickled before the field existed reads as known only when ``MSM.linear`` built it,
  by the exact type of its design;
* every fit entry refuses a restored or modified model before any learner or design call;
* a result restored with an undeclared written design keeps its stored estimates, and every
  recomputation from it refuses, while a restored ``MSM.linear`` result still recomputes;
* the simulated-confounding replay refuses such a result before it runs the design or the
  weight, and carries the declaration of a model it admits;
* a deliberate mutation that removes a refusal makes those witnesses fail;
* on an exact law, a design centred at the sample mean of ``W`` and declared ``"known"`` gets
  the fixed-centre curve, which understates the standard error of the intercept;
* a design with a fixed centre, and ``MSM.linear``, keep their intervals.

The weight twin of the replay witness, a restored result that lost ``weights_kind`` on a
declared design, is ``tests/unit/test_msm_projection_weights.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from typing import Any, ClassVar

import numpy as np
import pytest

import cleverly.longitudinal.estimator as ltmle_module
import cleverly.msm as msm_module
from cleverly import CausalStudy, MSMProjection, PointTreatment
from cleverly.data import CausalData
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError, DataError
from cleverly.msm import (
    _ESTIMATED_DESIGN,
    _UNDECLARED_DESIGN,
    MSM,
    _LinearDesign,
    refuse_msm_functions,
)
from cleverly.sensitivity import _simulated_confounding_fixed as replay_module
from cleverly.sensitivity import simulated_confounding
from tests import discrete_law as law
from tests.conftest import OracleOutcome, OracleTreatment, linear_in_sample
from tests.pickles import legacy_without
from tests.unit._confounding_support import forbid_draw_and_refit
from tests.unit._declaration_support import (
    PATHWISE,
    assert_every_witness_fails,
    assert_refused,
    assert_refused_before_any_call,
    gateaux_eif,
    recomputations,
    restored,
    restored_states,
    se_ratio,
    tmle_module,
)
from tests.unit._declaration_support import legacy_result as legacy_result_of
from tests.unit._natural_course_support import NeverFit, never_fit_learners
from tests.unit.test_msm_projection_weights import (
    DOSE_SLOPE,
    counted_msm_fit,
    dose_surface,
    duration_design,
    linear,
    ltmle_fit,
    tmle_fit,
    uniform_dose_fit,
    validate_replay,
)
from tests.unit.test_simulated_confounding_policies import (
    _GRID,
    _alias,
    _estimate,
    _fit_msm,
    _study,
)

#: The two declaration refusals, imported from the module that raises them, so the text
#: is written once.
UNDECLARED = _UNDECLARED_DESIGN
ESTIMATED = _ESTIMATED_DESIGN
#: Fragments of the other refusals.  A test matches a fragment, not the whole text, so a
#: rewording of the explanation does not break it, but a message from another check does.
UNKNOWN = "design_kind must be 'known', 'estimated' or None"
SIGNATURES = (
    "MSM design= must be a callable",
    "(arm_label, covariate_frame) -> (n, p) for a point treatment",
    "(regimen_label, horizon, baseline_frame) -> (n, p) for a longitudinal regimen MSM",
)


@dataclass(frozen=True)
class FixedDesign:
    """A known design, ``[1, a, W]``.  A class rather than a lambda, so a model pickles."""

    def __call__(self, arm: Any, frame: Any) -> np.ndarray:
        n = len(frame)
        return np.column_stack(
            [np.ones(n), np.full(n, float(arm)), np.asarray(frame["W"], dtype=float)]
        )


class SpyDesign:
    """A known design that counts its calls.  A refusal must come before the first one.

    It is full rank, ``[1, a]`` for an arm and ``[1, duration]`` for a regimen, because a
    valid fit evaluates the design and checks its rank before the first learner.
    """

    calls: ClassVar[int] = 0

    def __call__(self, *arguments: Any) -> np.ndarray:
        type(self).calls += 1
        if len(arguments) == 3:
            return duration_design(*arguments)
        arm, frame = arguments
        return np.column_stack([np.ones(len(frame)), np.full(len(frame), float(arm))])


def written(design: Any = None, **declaration: Any) -> MSM:
    """A working model with a written design, ``[1, a, W]`` unless ``design`` is given."""
    return MSM(
        design=FixedDesign() if design is None else design, terms=law.MSM_TERMS, **declaration
    )


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
        assert_refused(lambda: written(design_kind=kind), DataError, UNKNOWN)

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

    def test_the_shorthand_declares_its_own_design_known(self) -> None:
        assert linear().design_kind == "known"
        assert MSM.linear().design_kind == "known"

    def test_the_declaration_is_the_last_field(self) -> None:
        """The full order is pinned by the RM13 module, which added the field before it."""
        assert [field.name for field in fields(MSM)][-2:] == ["weights_kind", "design_kind"]


# ------------------------------------------------------------------ old pickles


class TestTheLegacyRuleReadsTheExactType:
    """A model saved before the field existed reads known only if ``MSM.linear`` built it."""

    def test_a_legacy_shorthand_model_reads_known(self) -> None:
        old = legacy_without(linear(), "design_kind")
        assert "design_kind" not in vars(old)
        assert old.design_kind is None
        assert refuse_msm_functions(old) is None
        assert replace(old).design_kind is None
        assert refuse_msm_functions(restored(linear(), "design_kind", None)) is None

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
    SpyDesign.calls = 0
    return MSM(design=SpyDesign(), terms=("(intercept)", "a"), design_kind="known")


def regimen_model() -> MSM:
    SpyDesign.calls = 0
    return MSM(design=SpyDesign(), terms=("(intercept)", "duration"), design_kind="known")


def study_fit(model: MSM, learners: dict[str, Any]) -> Any:
    study = CausalStudy(
        law.frame(), design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",))
    )
    return study.identify(MSMProjection(model)).estimate(
        outcome_learner=learners["outcome_learner"],
        treatment_learner=learners["treatment_learner"],
        cross_fit=False,
        simultaneous=False,
    )


def direct_refit(model: MSM, learners: dict[str, Any]) -> Any:
    """``TMLE.refit`` on prepared data: the entry the refutations and the replay use."""
    estimator = TMLE(msm=model, cross_fit=False, simultaneous=False, **learners)
    return estimator.refit(CausalData.from_frame(law.frame(), outcome="Y", treatment="A"))


#: Every fit entry that can reach a restored model: its model builder and the fit.  Each
#: fit resets ``NeverFit``.
ENTRIES: dict[str, tuple[Callable[[], MSM], Callable[[MSM], Any]]] = {
    "fit": (point_model, lambda model: tmle_fit(model, never_fit_learners())),
    "study": (point_model, lambda model: study_fit(model, never_fit_learners())),
    "refit": (point_model, lambda model: direct_refit(model, never_fit_learners())),
    "ltmle": (regimen_model, ltmle_fit),
}

#: What a restored model can carry, and the refusal each one meets.
RESTORED = restored_states(UNDECLARED, ESTIMATED)


def assert_entry_refuses(entry: str, kind: Any, *fragments: str) -> None:
    build, fit = ENTRIES[entry]
    model = restored(build(), "design_kind", kind)
    assert_refused_before_any_call(lambda: fit(model), SpyDesign, "design", *fragments)


class TestTheFitRefusesARestoredModel:
    @pytest.mark.parametrize("entry", list(ENTRIES))
    @pytest.mark.parametrize("name", list(RESTORED))
    def test_every_entry_refuses_before_any_learner_or_design_call(
        self, entry: str, name: str
    ) -> None:
        kind, fragments = RESTORED[name]
        assert_entry_refuses(entry, kind, *fragments)

    @pytest.mark.parametrize("entry", list(ENTRIES))
    def test_a_known_declaration_reaches_the_first_learner(self, entry: str) -> None:
        """The control: the same entries with the declaration intact go past the check."""
        build, fit = ENTRIES[entry]
        model = build()
        with pytest.raises(AssertionError, match="before any learner is fitted"):
            fit(model)
        assert NeverFit.calls == 1


# ------------------------------------------------------------------ a restored result


def legacy_result(result: Any) -> Any:
    """``result`` as an artifact written before ``design_kind`` existed would restore it."""
    return legacy_result_of(result, "design_kind", lambda estimator: [estimator.msm])


def in_sample_fit(model: MSM) -> Any:
    estimator = TMLE(msm=model, **linear_in_sample())
    return estimator.fit(law.frame(), outcome="Y", treatment="A").single()


#: The estimands a retarget of the legacy result requests.
RETARGETED = ("msm",)
RECOMPUTATIONS = ["truncation_curve", "retarget", "refit"]


class TestALegacyResultKeepsItsNumbersAndRefusesARecomputation:
    """RM27: a restored result with a written design holds what it computed.

    Loading checks nothing, so the stored estimates answer as they were saved.  Every sweep
    recomputes through ``_retarget_detailed``, and a refit through
    ``_resolve_estimands_for_data``, and both check the declaration as the fit does.
    """

    @pytest.fixture(scope="class")
    def result(self) -> Any:
        return in_sample_fit(written(design_kind="known"))

    @pytest.fixture(scope="class")
    def shorthand(self) -> Any:
        return in_sample_fit(linear())

    def test_the_stored_interval_answers_unchanged(self, result: Any) -> None:
        old = legacy_result(result)
        assert "msm[W]" in result.estimates
        for name, estimate in result.estimates.items():
            assert old[name].ci == estimate.ci
            assert old[name].psi == estimate.psi

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
        old = legacy_result_of(
            legacy_result(shorthand), "weights_kind", lambda estimator: [estimator.msm]
        )
        assert type(old.estimator.msm.design) is _LinearDesign
        recomputations(old, RETARGETED)[entry]()


# ------------------------------------------------------------------ the replay


def shorthand_fit() -> Any:
    """A replayable fit of ``MSM.linear``, with uniform weights."""
    return _estimate(_study(), MSMProjection(MSM.linear(modifiers=("W",), interaction=False)))


def legacy_shorthand_fit() -> Any:
    return legacy_result(shorthand_fit())


#: The coefficient of the shorthand fit that the replay targets.
SLOPE = "a"


class TestTheReplayChecksAndCarriesTheDeclaration:
    def test_a_legacy_written_design_refuses_before_any_user_function_runs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        old, design, weights = counted_msm_fit("design_kind")
        forbid_draw_and_refit(monkeypatch, old.estimator)
        assert_refused(
            lambda: simulated_confounding(old, estimand=_alias(old), grid=_GRID, random_state=31),
            CapabilityError,
            UNDECLARED,
        )
        assert_refused(lambda: validate_replay(old), CapabilityError, UNDECLARED)
        assert design.calls == 0, "the design was evaluated before the refusal"
        assert weights.calls == 0, "the weight was evaluated before the refusal"

    def test_removing_the_replay_check_evaluates_both_and_still_refuses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation M3: without the check the replay runs the design and the weight.

        ``replace`` still refuses the undeclared design, because the replay passes the
        source model's declaration and never forges ``"known"``.
        """
        old, design, weights = counted_msm_fit("design_kind")
        monkeypatch.setattr(replay_module, "refuse_msm_functions", lambda model: None)
        assert_refused(lambda: validate_replay(old), CapabilityError, UNDECLARED)
        assert design.calls == 2
        assert weights.calls == 2

    def test_a_known_written_design_replays_declared(self) -> None:
        assert validate_replay(_fit_msm()).msm.design_kind == "known"

    def test_a_legacy_shorthand_fit_replays(self) -> None:
        """The frozen arrays replace ``_LinearDesign``, so the replay carries the rule."""
        result = shorthand_fit()
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

        A declared source model would carry ``"known"`` through ``replace`` on its own, so
        the witness is the legacy shorthand fit, whose declaration only the replay supplies.
        The RM13 module holds the same test for the weight.
        """
        old = legacy_shorthand_fit()

        def dropping(model: Any, **changes: Any) -> Any:
            changes.pop("design_kind", None)
            return replace(model, **changes)

        monkeypatch.setattr(replay_module, "replace", dropping)
        assert_refused(lambda: validate_replay(old, SLOPE), CapabilityError, UNDECLARED)


# ------------------------------------------------------------------ mutation controls


def declaration_witnesses() -> list[Callable[[], None]]:
    """Every declaration-layer witness above, as a call that must raise to pass."""
    suite = TestTheDeclarationIsRequired()
    legacy = TestTheLegacyRuleReadsTheExactType()
    return [
        suite.test_an_undeclared_design_is_refused,
        suite.test_an_estimated_design_is_refused_by_its_missing_term,
        lambda: suite.test_an_unknown_declaration_is_refused("Known"),
        lambda: suite.test_a_design_that_is_not_callable_is_refused(np.ones((3, 3)), "known"),
        legacy.test_a_forged_shorthand_flag_is_not_a_declaration,
        legacy.test_a_subclass_of_the_shorthand_design_is_not_a_declaration,
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
        """Mutation M2, at the fit."""
        kind, fragments = RESTORED[name]
        monkeypatch.setattr(tmle_module, "refuse_msm_functions", lambda model: None)
        monkeypatch.setattr(ltmle_module, "refuse_msm_functions", lambda model: None)
        with pytest.raises(AssertionError):
            assert_entry_refuses(entry, kind, *fragments)
        assert NeverFit.calls > 0, "the mutated fit refused before a learner"

    @pytest.mark.parametrize("entry", RECOMPUTATIONS)
    def test_removing_the_fit_layer_check_fails_the_recomputation_refusal(
        self, entry: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation M2, at a recomputation of a legacy result."""
        old = legacy_result(in_sample_fit(written(design_kind="known")))
        monkeypatch.setattr(tmle_module, "refuse_msm_functions", lambda model: None)
        with pytest.raises(AssertionError):
            assert_refused(recomputations(old, RETARGETED)[entry], CapabilityError, UNDECLARED)

    def test_every_site_calls_the_one_refusal(self) -> None:
        """One refusal, one text: the model, the fit layer and the replay share one check."""
        assert msm_module.refuse_msm_functions is refuse_msm_functions
        assert tmle_module.refuse_msm_functions is refuse_msm_functions
        assert ltmle_module.refuse_msm_functions is refuse_msm_functions
        assert replay_module.refuse_msm_functions is refuse_msm_functions
        assert replay_module._design_kind is msm_module._design_kind


# ------------------------------------------------------------------ the witness

#: The law of the witness.  ``Qbar`` rises in ``W`` in both arms, so the ``W`` coefficient
#: is material, and every cell is a multiple of ``1 / N``, so the 1000 rows realise the law
#: exactly.  ``P(W)`` is the default, whose mean of ``W`` is 0.7.
CENTRE_COUNTS = law.cell_counts(q=[[0.1, 0.2], [0.5, 0.6], [0.8, 0.9]])
CENTRE_PROBS = CENTRE_COUNTS / law.N
CELL_P = np.array([CENTRE_PROBS[cell] for cell in law.SUPPORT])
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
    return np.asarray(gateaux_eif(lambda p: law.msm_beta(p, UNIFORM, design=design), CENTRE_PROBS))


class SampleCentre:
    """The user lie: ``W`` centred at its sample mean, declared ``"known"``."""

    def __init__(self, w: Any) -> None:
        self.centre = float(np.mean(w))

    def __call__(self, arm: Any, frame: Any) -> np.ndarray:
        n = len(frame)
        w = np.asarray(frame["W"], dtype=float)
        return np.column_stack([np.ones(n), np.full(n, float(arm)), w - self.centre])


class FixedCentre(SampleCentre):
    """``W`` centred at a stated constant: a known design."""

    def __init__(self, centre: float) -> None:
        self.centre = centre


def oracle_fit(model: MSM) -> Any:
    """In sample, with the law's own nuisances, so the curve has no learner error."""
    dgp = law.DiscreteLaw(CENTRE_PROBS)
    estimator = TMLE(
        outcome_learner=OracleOutcome(dgp),
        treatment_learner=OracleTreatment(dgp),
        cross_fit=False,
        msm=model,
        estimands="all",
        simultaneous=False,
        random_state=0,
    )
    return estimator.fit(law.frame(CENTRE_COUNTS), outcome="Y", treatment="A").single()


def curve(fit: Any, index: int) -> np.ndarray:
    return np.asarray(fit.estimates[f"msm[{law.MSM_TERMS[index]}]"].influence_curve)


@pytest.fixture(scope="module")
def centre_fit() -> Any:
    sample = law.frame(CENTRE_COUNTS)
    return oracle_fit(written(SampleCentre(sample["W"]), design_kind="known"))


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
        assert SampleCentre(law.frame(CENTRE_COUNTS)["W"]).centre == pytest.approx(
            CENTRE, abs=1e-12
        )
        beta = law.msm_beta(CENTRE_PROBS, UNIFORM, design=FIXED_DESIGN)
        for index, term in enumerate(law.MSM_TERMS):
            assert centre_fit.estimates[f"msm[{term}]"].psi == pytest.approx(beta[index], abs=1e-12)

    @pytest.mark.parametrize("index", range(len(law.MSM_TERMS)))
    def test_control_the_curve_is_the_fixed_centre_eif(self, centre_fit: Any, index: int) -> None:
        """Control: the curve is right for the functional the user declared."""
        np.testing.assert_allclose(
            curve(centre_fit, index)[law.first_row_of(CENTRE_COUNTS)],
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
        ratio = se_ratio(curve(centre_fit, INTERCEPT), design_eif(centred)[:, INTERCEPT], CELL_P)
        assert ratio == pytest.approx(0.8927, abs=1e-4)
        assert ratio < UNDERSTATEMENT_BOUND

    @pytest.mark.parametrize("term", ["a", "W"])
    def test_control_the_other_coefficients_are_unaffected(
        self, centre_fit: Any, term: str
    ) -> None:
        """Control: the centre shifts the intercept only, so ``a`` and ``W`` read 1."""
        index = law.MSM_TERMS.index(term)
        ratio = se_ratio(curve(centre_fit, index), design_eif(centred)[:, index], CELL_P)
        assert abs(ratio - 1.0) < 1e-9

    def test_mutation_a_frozen_oracle_loses_the_witness(self, centre_fit: Any) -> None:
        """Mutation M6: an oracle that froze the centre would read 1 and fail the bound."""
        frozen = se_ratio(
            curve(centre_fit, INTERCEPT), design_eif(FIXED_DESIGN)[:, INTERCEPT], CELL_P
        )
        assert frozen == pytest.approx(1.0, abs=1e-9)
        assert not frozen < UNDERSTATEMENT_BOUND

    def test_an_estimated_declaration_is_refused_on_this_law(self) -> None:
        """The honest declaration of the sample centre meets the refusal, before any fit."""
        centre = SampleCentre(law.frame(CENTRE_COUNTS)["W"])
        assert_refused(lambda: written(centre, design_kind="estimated"), CapabilityError, ESTIMATED)
        model = restored(written(centre, design_kind="known"), "design_kind", "estimated")
        assert_refused(lambda: tmle_fit(model, never_fit_learners()), CapabilityError, ESTIMATED)
        assert NeverFit.calls == 0


class TestAKnownDesignKeepsItsInterval:
    def test_a_fixed_centre_reports_the_fixed_centre_eif(self, centre_fit: Any) -> None:
        result = oracle_fit(written(FixedCentre(CENTRE), design_kind="known"))
        rows = law.first_row_of(CENTRE_COUNTS)
        for index, term in enumerate(law.MSM_TERMS):
            estimate = result.estimates[f"msm[{term}]"]
            assert estimate.inference == "influence_curve"
            assert np.all(np.isfinite(estimate.ci))
            np.testing.assert_allclose(
                curve(result, index)[rows], design_eif(FIXED_DESIGN)[:, index], atol=1e-10, rtol=0
            )
            np.testing.assert_array_equal(curve(result, index), curve(centre_fit, index))

    def test_the_shorthand_reports_its_eif(self) -> None:
        result = oracle_fit(linear())
        rows = law.first_row_of(CENTRE_COUNTS)
        for index, term in enumerate(law.MSM_TERMS):
            estimate = result.estimates[f"msm[{term}]"]
            assert estimate.inference == "influence_curve"
            assert np.all(np.isfinite(estimate.ci))
            np.testing.assert_allclose(
                curve(result, index)[rows], design_eif(None)[:, index], atol=1e-10, rtol=0
            )
