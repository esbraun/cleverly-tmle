r"""An MSM projection weight must be declared known, and an estimated one is refused.

The projection weight :math:`h(a, V)` is part of the estimand.  When it is a fixed function,
the influence curve the package reports is the efficient influence function of
:math:`\beta`.  When it is computed from the sample, such as the arm shares or a fitted
mechanism, :math:`h` is a functional of :math:`P`.  The efficient influence function then
carries a further term for the pathwise derivative of :math:`h`, the reported curve does not
have it, and the reported standard error is too small.

A callable can close over any estimate, and no code can inspect a closure, so the status of
the weight is a declaration: ``MSM(weights_kind=...)``.  RM13 in ``docs/roadmap.md`` records
the defect.  This module pins seven things:

* the declaration is required, and ``"estimated"`` is refused, with their messages;
* the fit refuses a restored or modified model before any learner or weight call;
* ``MSMSet.evaluate`` and ``evaluate_regimen_msm``, called directly, refuse such a model
  before the weight runs;
* a model pickled before the field existed loads, and a uniform-weight fit still replays;
* a result restored with an undeclared callable weight keeps its stored estimates, and
  every recomputation from it refuses;
* a deliberate mutation that removes the refusal makes those witnesses fail;
* on an exact law, a share weight declared ``"known"`` gets an influence curve that is right
  for the fixed-weight functional and understates the variance of the estimated-weight one.

The replay refuses a restored undeclared weight before it runs the design or the weight.
``tests/unit/test_msm_design_declaration.py`` drives that witness for both MSM
declarations, and ``tests/unit/_msm_declaration_support.py`` holds the builders the two
files share.

The last item is the reason for the refusal, measured without sampling error.  The testing
strategy keeps repeated sampling out of the fast suite, and the exact-law gap is the
asymptotic form of "the reported standard error is below the sampling standard deviation".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields, replace
from typing import Any

import numpy as np
import pytest

import cleverly.msm as msm_module
from cleverly.assessment import replayability
from cleverly.exceptions import CapabilityError, DataError
from cleverly.msm import (
    _ESTIMATED_WEIGHTS,
    _UNDECLARED_WEIGHTS,
    MSM,
    refuse_unsupported,
)
from cleverly.sensitivity import simulated_confounding
from tests import discrete_law as law
from tests.pickles import legacy_without
from tests.unit._confounding_support import Counter, validate_replay
from tests.unit._declaration_support import (
    PATHWISE,
    assert_every_witness_fails,
    assert_keeps_its_interval,
    assert_refused,
    assert_refused_before_any_call,
    assert_replay_agrees,
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
    DOSE_SLOPE,
    WEIGHTS_UNKNOWN,
    FixedWeight,
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
)
from tests.unit._natural_course_support import NeverFit, never_fit_learners
from tests.unit._simulated_confounding_support import _GRID, _alias, _fit_msm

#: The two declaration refusals, imported from the module that raises them, so the text
#: is written once.
UNDECLARED = _UNDECLARED_WEIGHTS
ESTIMATED = _ESTIMATED_WEIGHTS
#: Fragments of the other refusals.  A test matches a fragment, not the whole text, so a
#: rewording of the explanation does not break it, but a message from another check does.
NOT_CALLABLE = "must be a callable"


def unit_weight(*arguments: Any) -> np.ndarray:
    """A known weight of 1 for either signature.  A ``Counter`` around it counts its calls."""
    return np.ones(len(arguments[-1]))


# ------------------------------------------------------------------ the declaration


class TestTheDeclarationIsRequired:
    def test_an_undeclared_callable_is_refused(self) -> None:
        assert_refused(
            lambda: linear(weights=FixedWeight()), CapabilityError, UNDECLARED, "'known'"
        )

    def test_an_estimated_weight_is_refused_by_its_missing_term(self) -> None:
        assert_refused(
            lambda: linear(weights=FixedWeight(), weights_kind="estimated"),
            CapabilityError,
            ESTIMATED,
            PATHWISE,
            "functional of P",
        )

    @pytest.mark.parametrize("kind", [None, "known", "estimated"])
    def test_an_array_is_refused_whatever_it_declares(self, kind: Any) -> None:
        """An array is one evaluation of ``h``, and nothing shows it was not estimated."""
        assert_refused(
            lambda: linear(weights=np.ones(10), weights_kind=kind),
            CapabilityError,
            NOT_CALLABLE,
            PATHWISE,
        )

    def test_the_array_refusal_names_both_callable_signatures(self) -> None:
        """A point-treatment and a regimen MSM take different callables, and both refuse."""
        signatures = (
            "(arm_label, covariate_frame) -> (n,) for a point treatment",
            "(regimen_label, horizon, baseline_frame) -> (n,) for a longitudinal regimen MSM",
        )
        assert_refused(
            lambda: linear(weights=np.ones(10), weights_kind="known"),
            CapabilityError,
            *signatures,
        )
        assert_refused(
            lambda: MSM(
                design=duration_design,
                terms=("(intercept)", "duration"),
                weights=np.ones(10),
                weights_kind="known",
                design_kind="known",
            ),
            CapabilityError,
            *signatures,
        )

    def test_an_estimated_declaration_without_a_weight_is_inconsistent(self) -> None:
        assert_refused(lambda: linear(weights_kind="estimated"), DataError, "this model has none")

    @pytest.mark.parametrize("kind", ["Known", "probability", True, 1])
    def test_an_unknown_declaration_is_refused(self, kind: Any) -> None:
        assert_refused(
            lambda: linear(weights=FixedWeight(), weights_kind=kind), DataError, WEIGHTS_UNKNOWN
        )

    def test_a_known_callable_is_accepted_and_forwarded_by_the_shorthand(self) -> None:
        model = linear(weights=FixedWeight(), weights_kind="known")
        assert model.weights_kind == "known"
        assert isinstance(model.weights, FixedWeight)

    @pytest.mark.parametrize("kind", [None, "known"])
    def test_uniform_weights_need_no_declaration(self, kind: Any) -> None:
        """Uniform weights are known, so no declaration is needed for them."""
        assert linear(weights_kind=kind).weights is None
        model = MSM(design=lambda a, w: np.ones((len(w), 1)), terms=("c",), design_kind="known")
        assert model.weights_kind is None

    def test_the_named_refusal_is_a_capability_error(self) -> None:
        assert_refused(
            lambda: refuse_unsupported("estimated_weights"), CapabilityError, ESTIMATED, PATHWISE
        )


class TestThePositionalOrderIsUnchanged:
    """The declarations are the last fields, so ``link`` stays the fourth positional argument.

    ``weights_kind`` follows ``doses``, and ``design_kind`` (RM27) follows ``weights_kind``.
    """

    def test_the_declaration_is_the_last_field(self) -> None:
        assert [field.name for field in fields(MSM)] == [
            "design",
            "terms",
            "weights",
            "link",
            "from_linear",
            "doses",
            "weights_kind",
            "design_kind",
        ]

    def test_the_fourth_positional_argument_is_the_link(self) -> None:
        model = MSM(duration_design, ("(intercept)", "duration"), None, "log", design_kind="known")
        assert model.link == "log"
        assert model.weights_kind is None
        declared = MSM(
            duration_design,
            ("(intercept)", "duration"),
            FixedWeight(),
            "log",
            weights_kind="known",
            design_kind="known",
        )
        assert declared.link == "log"
        assert declared.weights_kind == "known"

    def test_a_positional_weight_without_a_declaration_still_refuses(self) -> None:
        assert_refused(
            lambda: MSM(
                duration_design,
                ("(intercept)", "duration"),
                FixedWeight(),
                "log",
                design_kind="known",
            ),
            CapabilityError,
            UNDECLARED,
        )


# ------------------------------------------------------------------ the fit layer


def point_model() -> MSM:
    """A point MSM whose known weight counts its calls in ``model.weights.calls``."""
    return linear(weights=Counter(unit_weight), weights_kind="known")


def regimen_model() -> MSM:
    """A regimen MSM whose known weight counts its calls in ``model.weights.calls``."""
    return MSM(
        design=duration_design,
        terms=("(intercept)", "duration"),
        weights=Counter(unit_weight),
        weights_kind="known",
        design_kind="known",
    )


def assert_tmle_refuses(kind: Any, *fragments: str) -> None:
    model = restored(point_model(), "weights_kind", kind)
    assert_refused_before_any_call(
        lambda: tmle_fit(model, never_fit_learners()), model.weights, "weight", *fragments
    )


def assert_ltmle_refuses(kind: Any, *fragments: str) -> None:
    model = restored(regimen_model(), "weights_kind", kind)
    assert_refused_before_any_call(lambda: ltmle_fit(model), model.weights, "weight", *fragments)


#: What a restored model can carry, and the refusal each one meets.
RESTORED = restored_states(UNDECLARED, ESTIMATED)


class TestTheFitRefusesARestoredUndeclaredModel:
    @pytest.mark.parametrize("name", list(RESTORED))
    def test_tmle_refuses_before_any_learner_or_weight_call(self, name: str) -> None:
        kind, fragments = RESTORED[name]
        assert_tmle_refuses(kind, *fragments)

    @pytest.mark.parametrize("name", list(RESTORED))
    def test_ltmle_refuses_before_any_learner_or_weight_call(self, name: str) -> None:
        kind, fragments = RESTORED[name]
        assert_ltmle_refuses(kind, *fragments)

    def test_a_known_declaration_reaches_the_first_learner(self) -> None:
        """The control: the same fits with the declaration intact go past the check."""
        with pytest.raises(AssertionError, match="before any learner is fitted"):
            tmle_fit(point_model(), never_fit_learners())
        assert NeverFit.calls == 1
        with pytest.raises(AssertionError, match="before any learner is fitted"):
            ltmle_fit(regimen_model())
        assert NeverFit.calls == 1


# ------------------------------------------------------------------ the evaluators

#: Each public evaluator that runs a working model's design and weight, with a builder of a
#: model it accepts.  ``tests/unit/test_msm_design_declaration.py`` drives the same pair.
EVALUATORS: dict[str, tuple[Callable[[], MSM], Callable[[MSM], Any]]] = {
    "MSMSet.evaluate": (point_model, evaluate_point),
    "evaluate_regimen_msm": (regimen_model, evaluate_regimen),
}


class TestTheEvaluatorsCheckFirst:
    """A restored model handed straight to an evaluator refuses before its weight runs."""

    @pytest.mark.parametrize("evaluator", list(EVALUATORS))
    @pytest.mark.parametrize("name", list(RESTORED))
    def test_a_restored_model_refuses_before_the_weight_runs(
        self, evaluator: str, name: str
    ) -> None:
        kind, fragments = RESTORED[name]
        build, evaluate = EVALUATORS[evaluator]
        model = restored(build(), "weights_kind", kind)
        assert_refused(lambda: evaluate(model), CapabilityError, *fragments)
        assert model.weights.calls == 0, "the weight was evaluated before the refusal"

    @pytest.mark.parametrize("evaluator", list(EVALUATORS))
    def test_a_declared_model_is_evaluated(self, evaluator: str) -> None:
        """The control: the same call with the declaration intact runs the weight."""
        build, evaluate = EVALUATORS[evaluator]
        model = build()
        evaluate(model)
        assert model.weights.calls > 0


# ------------------------------------------------------------------ old pickles


def legacy_result(result: Any) -> Any:
    """``result`` as an artifact written before ``weights_kind`` existed would restore it."""
    return legacy_msm_result(result, "weights_kind")


class TestALegacyModelLoads:
    def test_a_pickle_without_the_field_reads_none_and_can_be_replaced(self) -> None:
        old = legacy_without(linear(), "weights_kind")
        assert "weights_kind" not in vars(old)
        assert old.weights_kind is None
        assert replace(old).weights_kind is None
        assert replace(old, weights_kind="known").weights_kind == "known"
        assert replace(old, weights=FixedWeight(), weights_kind="known").weights_kind == "known"

    def test_a_legacy_callable_weight_refuses_at_the_fit(self) -> None:
        """An old model with a callable weight has no declaration, so it is refused."""
        old = legacy_without(linear(weights=FixedWeight(), weights_kind="known"), "weights_kind")
        assert old.weights_kind is None
        assert_refused(lambda: tmle_fit(old, never_fit_learners()), CapabilityError, UNDECLARED)
        assert NeverFit.calls == 0
        assert_refused(lambda: replace(old), CapabilityError, UNDECLARED)

    def test_a_legacy_uniform_weight_fit_replays(self) -> None:
        """``_freeze_msm`` swaps the uniform weight for frozen arrays, declared known."""
        result = _fit_msm(saturated=True)
        assert result.estimator.msm.weights is None
        alias = _alias(result)
        expected = simulated_confounding(result, estimand=alias, grid=_GRID, random_state=31)
        old = legacy_result(result)
        assert "weights_kind" not in vars(old.estimator.msm)
        surface = simulated_confounding(old, estimand=alias, grid=_GRID, random_state=31)
        assert all(cell.failure is None for cell in surface.cells)
        assert surface == expected
        assert validate_replay(old).msm.weights_kind == "known"

    def test_a_uniform_weight_dose_fit_replays(self) -> None:
        """The continuous twin: ``_freeze_msm`` declares the frozen dose weight known.

        The model declares no weight, so only the replay can supply ``"known"``.  Every
        other continuous replay declares its weight, which ``replace`` copies on its own.
        """
        result = uniform_dose_fit()
        assert result.estimator.msm.weights is None
        assert result.estimator.msm.weights_kind is None
        dose_surface(result)
        assert validate_replay(result, DOSE_SLOPE).msm.weights_kind == "known"

    def test_a_replay_that_drops_the_declaration_refuses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The deliberate-mutation control for the replay: drop what ``_freeze_msm`` passes."""
        result = _fit_msm(saturated=True)
        drop_from_the_replay(monkeypatch, "weights_kind")
        assert_refused(lambda: validate_replay(result), CapabilityError, UNDECLARED)


#: The estimands a retarget of the legacy result requests.
RETARGETED = ("msm",)


class TestALegacyResultKeepsItsPointEstimatesAndRefusesARecomputation:
    """RM13: a restored result keeps its point estimates and computes nothing new.

    Loading raises nothing.  A callable weight restored without its declaration gives the
    result the ``"undeclared_function_plugin"`` status of RM28, so the stored interval
    becomes a diagnostic.  Every sweep recomputes through ``_retarget_detailed``, which
    checks the declaration as the fit does.
    """

    @pytest.fixture(scope="class")
    def result(self) -> Any:
        """A fit whose model declares its callable weight known: the valid pre-load state."""
        return in_sample_fit(linear(weights=FixedWeight(), weights_kind="known"))

    def test_the_stored_interval_becomes_a_diagnostic(self, result: Any) -> None:
        old = legacy_result(result)
        assert_stored_interval_is_a_diagnostic(result, old)
        assert not replayability(old).retarget_cached_nuisances
        assert not replayability(old).refit_nuisances
        assert replayability(result).refit_nuisances

    def test_each_replay_slot_agrees_with_its_call(self, result: Any) -> None:
        """Both slots of the legacy result read false, and both calls refuse."""
        assert_replay_agrees(legacy_result(result), RETARGETED)
        assert_replay_agrees(result, RETARGETED)

    def test_a_legacy_uniform_weight_result_keeps_its_interval(self) -> None:
        """The over-refusal control: uniform weights are known, so no status applies."""
        result = in_sample_fit(linear())
        old = legacy_result(result)
        assert old.estimator.msm.weights is None
        assert_keeps_its_interval(result, old)

    @pytest.mark.parametrize("entry", ["truncation_curve", "retarget"])
    def test_every_recomputation_refuses(self, result: Any, entry: str) -> None:
        old = legacy_result(result)
        assert_refused(recomputations(old, RETARGETED)[entry], CapabilityError, UNDECLARED)

    @pytest.mark.parametrize("entry", ["truncation_curve", "retarget"])
    def test_the_declared_result_recomputes(self, result: Any, entry: str) -> None:
        """The control: the same entry on the result before the declaration was lost."""
        recomputations(result, RETARGETED)[entry]()

    @pytest.mark.parametrize("entry", ["truncation_curve", "retarget"])
    def test_removing_the_retarget_check_fails_the_refusal(
        self, result: Any, entry: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        old = legacy_result(result)
        monkeypatch.setattr(tmle_module, "refuse_msm_functions", lambda model: None)
        with pytest.raises(AssertionError):
            assert_refused(recomputations(old, RETARGETED)[entry], CapabilityError, UNDECLARED)


# ------------------------------------------------------------------ mutation controls


def declaration_witnesses() -> list[Callable[[], None]]:
    """Every declaration-layer witness above, as a call that must raise to pass."""
    suite = TestTheDeclarationIsRequired()
    return [
        suite.test_an_undeclared_callable_is_refused,
        suite.test_an_estimated_weight_is_refused_by_its_missing_term,
        lambda: suite.test_an_array_is_refused_whatever_it_declares("known"),
        suite.test_an_estimated_declaration_without_a_weight_is_inconsistent,
        lambda: suite.test_an_unknown_declaration_is_refused("Known"),
    ]


class TestTheWitnessesHaveTeeth:
    def test_removing_the_declaration_check_fails_every_declaration_witness(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(msm_module, "refuse_msm_functions", lambda model: None)
        assert_every_witness_fails(declaration_witnesses())

    @pytest.mark.parametrize("name", list(RESTORED))
    def test_removing_the_fit_layer_check_fails_the_fit_witnesses(
        self, name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        kind, fragments = RESTORED[name]
        remove_every_fit_check(monkeypatch)
        with pytest.raises(AssertionError):
            assert_tmle_refuses(kind, *fragments)
        assert NeverFit.calls > 0, "the mutated TMLE fit refused before a learner"
        with pytest.raises(AssertionError):
            assert_ltmle_refuses(kind, *fragments)
        assert NeverFit.calls > 0, "the mutated LTMLE fit refused before a learner"


# ------------------------------------------------------------------ the witness

#: A law on which a share weight is not a constant of the design.  ``g = 0.5`` in every
#: stratum makes the arm share 0.5, and ``Qbar`` differs across ``W`` so that the ``W``
#: coefficient reads the arm weights.  Every cell is a multiple of ``1 / N``.
SHARE_COUNTS = law.cell_counts(
    p_w=[0.4, 0.2, 0.4], g=[0.5, 0.5, 0.5], q=[[0.05, 0.05], [0.3, 0.7], [0.05, 0.95]]
)
SHARE_PROBS = SHARE_COUNTS / law.N
CELL_P = cell_p(SHARE_PROBS)


def arm_share(probs: Any) -> Any:
    """``h(a, W = w) = P(A = a)``, a functional of the law, as a ``(3, 2)`` array."""
    p = np.asarray(probs)
    return np.ones((3, 1)) * p.sum(axis=(0, 2))[None, :]


#: The share at the law, frozen: the weight the user declared as known.
FROZEN_SHARE = arm_share(SHARE_PROBS)


class SampleShare:
    """The user lie: a weight read off the sample's arm shares, declared ``"known"``."""

    def __init__(self, treatment: Any) -> None:
        self.treated = float(np.mean(treatment))

    def __call__(self, arm: Any, frame: Any) -> np.ndarray:
        share = self.treated if float(arm) == 1.0 else 1.0 - self.treated
        return np.full(len(frame), share)


def eif(weights: Any) -> np.ndarray:
    """``(12, 3)``: the Gateaux derivative of ``law.msm_beta(P, weights)``, for every term.

    When ``weights`` is :func:`arm_share`, ``h`` is recomputed from the perturbed law, so the
    derivative carries the term through ``h`` that the fixed-weight derivative does not.
    """
    return msm_eif(SHARE_PROBS, weights)


#: The reported SE over the exact estimated-share SE for ``msm[W]`` must stay below this.
#: The measured ratio is 0.7417.  A reported curve that carried the share term would give
#: 1.  The bound sits between the two, far enough from 0.7417 that it does not pin a
#: digit, and far enough from 1 that an understatement of more than 20 percent is the claim.
UNDERSTATEMENT_BOUND = 0.8


@pytest.fixture(scope="module")
def share_fit() -> Any:
    weights = SampleShare(law.frame(SHARE_COUNTS)["A"])
    model = linear(weights=weights, weights_kind="known")
    return oracle_fit(SHARE_COUNTS, msm=model, estimands="all")


def share_se_ratio(fit: Any, index: int) -> float:
    """The reported curve's SE over the exact estimated-share EIF's SE.

    Both are asymptotic: the reported curve's second moment on the exact sample, and the
    estimated-share EIF's variance under the law.  ``std_error`` itself divides by
    ``n - 1``, which would move this by a factor of 1.0005.
    """
    return se_ratio(msm_curve(fit, index), eif(arm_share)[:, index], CELL_P)


class TestAnEstimatedShareWeightUnderstatesTheVariance:
    def test_the_frozen_share_is_the_fitted_weight(self, share_fit) -> None:
        """The premise: the sample share is exactly the law's share, 0.5 in both arms."""
        np.testing.assert_array_equal(FROZEN_SHARE, np.full((3, 2), 0.5))
        for index, term in enumerate(law.MSM_TERMS):
            assert share_fit.estimates[f"msm[{term}]"].psi == pytest.approx(
                law.msm_beta(SHARE_PROBS, FROZEN_SHARE)[index], abs=1e-12
            )

    @pytest.mark.parametrize("index", range(len(law.MSM_TERMS)))
    def test_control_the_curve_is_the_fixed_weight_eif(self, share_fit, index: int) -> None:
        """Control 1: the curve is right for the functional the user declared."""
        np.testing.assert_allclose(
            msm_curve(share_fit, index)[law.first_row_of(SHARE_COUNTS)],
            eif(FROZEN_SHARE)[:, index],
            atol=1e-10,
            rtol=0,
        )

    def test_the_oracle_carries_the_share_term(self) -> None:
        """The two oracles differ by exactly ``dbeta/dpi * (1{A = 1} - pi)``.

        ``dbeta/dpi`` comes from a complex step in the share alone, a path independent
        of the contamination path, so this checks that :func:`eif` differentiates
        through ``h`` and by the right amount.
        """
        step = 1e-30

        def at(pi: complex) -> Any:
            return np.ones((3, 1)) * np.array([1.0 - pi, pi])[None, :]

        slope = np.imag(law.msm_beta(SHARE_PROBS.astype(complex), at(0.5 + 1j * step))) / step
        np.testing.assert_allclose(slope, [-0.45, 0.0, 0.45], atol=1e-12)
        arm = np.array([a for _, a, _ in law.SUPPORT], dtype=float)
        np.testing.assert_allclose(
            eif(arm_share) - eif(FROZEN_SHARE), np.outer(arm - 0.5, slope), atol=1e-12, rtol=0
        )

    def test_witness_the_w_coefficient_understates_its_standard_error(self, share_fit) -> None:
        ratio = share_se_ratio(share_fit, law.MSM_TERMS.index("W"))
        assert ratio == pytest.approx(0.7417, abs=1e-4)
        assert ratio < UNDERSTATEMENT_BOUND

    def test_the_intercept_understates_its_standard_error(self, share_fit) -> None:
        ratio = share_se_ratio(share_fit, law.MSM_TERMS.index("(intercept)"))
        assert ratio == pytest.approx(0.9126, abs=1e-4)

    def test_control_the_arm_coefficient_is_unaffected(self, share_fit) -> None:
        """Control 2: on this design ``dbeta_a/dpi = 0``, so the term vanishes for ``a``."""
        assert share_se_ratio(share_fit, law.MSM_TERMS.index("a")) == pytest.approx(1.0, abs=1e-9)
