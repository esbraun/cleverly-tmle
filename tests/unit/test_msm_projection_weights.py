r"""An MSM projection weight must be declared known, and an estimated one is refused.

The projection weight :math:`h(a, V)` is part of the estimand.  When it is a fixed function,
the influence curve the package reports is the efficient influence function of
:math:`\beta`.  When it is computed from the sample, such as the arm shares or a fitted
mechanism, :math:`h` is a functional of :math:`P`.  The efficient influence function then
carries a further term for the pathwise derivative of :math:`h`, the reported curve does not
have it, and the reported standard error is too small.

A callable can close over any estimate, and no code can inspect a closure, so the status of
the weight is a declaration: ``MSM(weights_kind=...)``.  RM13 in ``docs/roadmap.md`` records
the defect.  This module pins six things:

* the declaration is required, and ``"estimated"`` is refused, with their messages;
* the fit refuses a restored or modified model before any learner or weight call;
* a model pickled before the field existed loads, and a uniform-weight fit still replays;
* a result restored with an undeclared callable weight keeps its stored estimates, and
  every recomputation from it refuses;
* a deliberate mutation that removes the refusal makes those witnesses fail;
* on an exact law, a share weight declared ``"known"`` gets an influence curve that is right
  for the fixed-weight functional and understates the variance of the estimated-weight one.

The last item is the reason for the refusal, measured without sampling error.  The testing
strategy keeps repeated sampling out of the fast suite, and the exact-law gap is the
asymptotic form of "the reported standard error is below the sampling standard deviation".
"""

from __future__ import annotations

import importlib
import pickle
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

import cleverly.longitudinal.estimator as ltmle_module
import cleverly.msm as msm_module
from cleverly.estimators import TMLE
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import CapabilityError, DataError
from cleverly.longitudinal import LTMLE
from cleverly.msm import (
    _ESTIMATED_WEIGHTS,
    _UNDECLARED_WEIGHTS,
    MSM,
    refuse_projection_weights,
    refuse_unsupported,
)
from cleverly.sensitivity import _simulated_confounding_fixed as replay_module
from cleverly.sensitivity import simulated_confounding
from cleverly.sensitivity.positivity import truncation_curve
from tests import discrete_law as law
from tests.conftest import OracleOutcome, OracleTreatment, linear_in_sample
from tests.unit._natural_course_support import NeverFit, never_fit_learners
from tests.unit.test_simulated_confounding_policies import _GRID, _alias, _fit_msm

#: ``cleverly.estimators`` exports a function named ``tmle``, which shadows the module.
tmle_module = importlib.import_module("cleverly.estimators.tmle")

#: The two declaration refusals, imported from the module that raises them, so the text
#: is written once.
UNDECLARED = _UNDECLARED_WEIGHTS
ESTIMATED = _ESTIMATED_WEIGHTS
#: Fragments of the other refusals.  A test matches a fragment, not the whole text, so a
#: rewording of the explanation does not break it, but a message from another check does.
NOT_CALLABLE = "must be a callable"
PATHWISE = "pathwise derivative"


@dataclass(frozen=True)
class FixedWeight:
    """A known weight, ``1 + a``.  A class rather than a lambda, so a model pickles."""

    def __call__(self, arm: Any, frame: Any) -> np.ndarray:
        return np.full(len(frame), 1.0 + float(arm))


class SpyWeight:
    """A known weight that counts its calls.  A refusal must come before the first one."""

    calls: ClassVar[int] = 0

    def __call__(self, *arguments: Any) -> np.ndarray:
        type(self).calls += 1
        frame = arguments[-1]
        return np.ones(len(frame))


def assert_refused(build: Callable[[], Any], error: type[Exception], *fragments: str) -> None:
    """``build()`` raises ``error`` with every fragment, or this raises ``AssertionError``.

    Written out rather than as ``pytest.raises``, whose failure is not an
    ``AssertionError``, so that the mutation controls below can require it to fail.
    """
    try:
        build()
    except error as raised:
        message = str(raised)
        for fragment in fragments:
            assert fragment in message, message
        return
    raise AssertionError(f"no {error.__name__} was raised")


def linear(**declaration: Any) -> MSM:
    return MSM.linear(modifiers=("W",), interaction=False, **declaration)


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
            ),
            CapabilityError,
            *signatures,
        )

    def test_an_estimated_declaration_without_a_weight_is_inconsistent(self) -> None:
        assert_refused(lambda: linear(weights_kind="estimated"), DataError, "this model has none")

    @pytest.mark.parametrize("kind", ["Known", "probability", True, 1])
    def test_an_unknown_declaration_is_refused(self, kind: Any) -> None:
        assert_refused(
            lambda: linear(weights=FixedWeight(), weights_kind=kind),
            DataError,
            "weights_kind must be 'known', 'estimated' or None",
        )

    def test_a_known_callable_is_accepted_and_forwarded_by_the_shorthand(self) -> None:
        model = linear(weights=FixedWeight(), weights_kind="known")
        assert model.weights_kind == "known"
        assert isinstance(model.weights, FixedWeight)

    @pytest.mark.parametrize("kind", [None, "known"])
    def test_uniform_weights_need_no_declaration(self, kind: Any) -> None:
        """Uniform weights are known, so no declaration is needed for them."""
        assert linear(weights_kind=kind).weights is None
        assert MSM(design=lambda a, w: np.ones((len(w), 1)), terms=("c",)).weights_kind is None

    def test_the_named_refusal_is_a_capability_error(self) -> None:
        assert_refused(
            lambda: refuse_unsupported("estimated_weights"), CapabilityError, ESTIMATED, PATHWISE
        )


class TestThePositionalOrderIsUnchanged:
    """``weights_kind`` is the last field, so ``link`` stays the fourth positional argument."""

    def test_the_declaration_is_the_last_field(self) -> None:
        assert [field.name for field in fields(MSM)] == [
            "design",
            "terms",
            "weights",
            "link",
            "from_linear",
            "doses",
            "weights_kind",
        ]

    def test_the_fourth_positional_argument_is_the_link(self) -> None:
        model = MSM(duration_design, ("(intercept)", "duration"), None, "log")
        assert model.link == "log"
        assert model.weights_kind is None
        declared = MSM(
            duration_design, ("(intercept)", "duration"), FixedWeight(), "log", weights_kind="known"
        )
        assert declared.link == "log"
        assert declared.weights_kind == "known"

    def test_a_positional_weight_without_a_declaration_still_refuses(self) -> None:
        assert_refused(
            lambda: MSM(duration_design, ("(intercept)", "duration"), FixedWeight(), "log"),
            CapabilityError,
            UNDECLARED,
        )


# ------------------------------------------------------------------ the fit layer


def restored(model: MSM, kind: Any) -> MSM:
    """``model`` with its declaration changed after construction, as a restore can leave it."""
    object.__setattr__(model, "weights_kind", kind)
    return model


def tmle_fit(model: MSM, learners: dict[str, Any]) -> Any:
    estimator = TMLE(msm=model, cross_fit=False, simultaneous=False, **learners)
    return estimator.fit(law.frame(), outcome="Y", treatment="A")


def panel(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Two nodes, censoring, and a binary end-of-study outcome."""
    rng = np.random.default_rng(seed)
    c1 = (rng.random(n) < 0.9).astype(float)
    c2 = np.where(c1 == 1, (rng.random(n) < 0.9).astype(float), np.nan)
    observed = (c1 == 1) & (c2 == 1)
    return pd.DataFrame(
        {
            "W1": rng.standard_normal(n),
            "A1": rng.integers(0, 2, n).astype(float),
            "C1": c1,
            "A2": np.where(c1 == 1, rng.integers(0, 2, n).astype(float), np.nan),
            "C2": c2,
            "Y": np.where(observed, rng.integers(0, 2, n).astype(float), np.nan),
        }
    )


def duration_design(label: Any, horizon: int, frame: Any) -> np.ndarray:
    del horizon
    return np.column_stack(
        [np.ones(len(frame)), np.full(len(frame), {"always": 2.0}.get(label, 0.0))]
    )


def ltmle_fit(model: MSM) -> Any:
    NeverFit.calls = 0
    estimator = LTMLE(
        {"always": 1, "never": 0},
        msm=model,
        n_folds=1,
        simultaneous=False,
        outcome_learner=NeverFit(),
        pseudo_learner=NeverFit(),
        treatment_learner=NeverFit(),
        censoring_learner=NeverFit(),
    )
    return estimator.fit(
        panel(),
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W1"],
        censoring=["C1", "C2"],
    )


def point_model() -> MSM:
    return linear(weights=SpyWeight(), weights_kind="known")


def regimen_model() -> MSM:
    return MSM(
        design=duration_design,
        terms=("(intercept)", "duration"),
        weights=SpyWeight(),
        weights_kind="known",
    )


def assert_tmle_refuses(kind: Any, *fragments: str) -> None:
    SpyWeight.calls = 0
    model = restored(point_model(), kind)
    assert_refused(lambda: tmle_fit(model, never_fit_learners()), CapabilityError, *fragments)
    assert NeverFit.calls == 0, f"{NeverFit.calls} learner fit(s) ran before the refusal"
    assert SpyWeight.calls == 0, "the weight was evaluated before the refusal"


def assert_ltmle_refuses(kind: Any, *fragments: str) -> None:
    SpyWeight.calls = 0
    model = restored(regimen_model(), kind)
    assert_refused(lambda: ltmle_fit(model), CapabilityError, *fragments)
    assert NeverFit.calls == 0, f"{NeverFit.calls} learner fit(s) ran before the refusal"
    assert SpyWeight.calls == 0, "the weight was evaluated before the refusal"


#: What a restored model can carry, and the refusal each one meets.
RESTORED = {"undeclared": (None, (UNDECLARED,)), "estimated": ("estimated", (ESTIMATED, PATHWISE))}


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


# ------------------------------------------------------------------ old pickles


def legacy(model: MSM) -> MSM:
    """``model`` as a pickle written before ``weights_kind`` existed would restore it."""
    copied = pickle.loads(pickle.dumps(model))
    vars(copied).pop("weights_kind")
    return pickle.loads(pickle.dumps(copied))


class TestALegacyModelLoads:
    def test_a_pickle_without_the_field_reads_none_and_can_be_replaced(self) -> None:
        old = legacy(linear())
        assert "weights_kind" not in vars(old)
        assert old.weights_kind is None
        assert replace(old).weights_kind is None
        assert replace(old, weights_kind="known").weights_kind == "known"
        assert replace(old, weights=FixedWeight(), weights_kind="known").weights_kind == "known"

    def test_a_legacy_callable_weight_refuses_at_the_fit(self) -> None:
        """An old model with a callable weight has no declaration, so it is refused."""
        old = legacy(linear(weights=FixedWeight(), weights_kind="known"))
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
        old = loads(dumps(result))
        vars(old.estimator.msm).pop("weights_kind")
        old = loads(dumps(old))
        assert "weights_kind" not in vars(old.estimator.msm)
        surface = simulated_confounding(old, estimand=alias, grid=_GRID, random_state=31)
        assert all(cell.failure is None for cell in surface.cells)
        assert surface == expected
        replay = replay_module.validate_fixed_replay(old, alias, old.parameter_keys[alias])
        assert replay.msm.weights_kind == "known"

    def test_a_replay_that_drops_the_declaration_refuses(self, monkeypatch) -> None:
        """The deliberate-mutation control for the replay: drop what ``_freeze_msm`` passes."""
        result = _fit_msm(saturated=True)

        def dropping(model: Any, **changes: Any) -> Any:
            changes.pop("weights_kind", None)
            return replace(model, **changes)

        monkeypatch.setattr(replay_module, "replace", dropping)
        alias = _alias(result)
        assert_refused(
            lambda: replay_module.validate_fixed_replay(
                result, alias, result.parameter_keys[alias]
            ),
            CapabilityError,
            UNDECLARED,
        )


def fitted_known() -> Any:
    """A fit whose model declares its callable weight known: the valid pre-load state."""
    return (
        TMLE(msm=linear(weights=FixedWeight(), weights_kind="known"), **linear_in_sample())
        .fit(law.frame(), outcome="Y", treatment="A")
        .single()
    )


def legacy_result(result: Any) -> Any:
    """``result`` as an artifact written before ``weights_kind`` existed would restore it."""
    old = loads(dumps(result))
    vars(old.estimator.msm).pop("weights_kind")
    old = loads(dumps(old))
    assert old.estimator.msm.weights_kind is None
    return old


def recomputations(result: Any) -> dict[str, Callable[[], Any]]:
    """Every entry to a recomputation a test drives: a sweep and the retarget it calls."""
    return {
        "truncation_curve": lambda: truncation_curve(result, bounds=[0.05]),
        "retarget": lambda: result.estimator.retarget(
            result.data, result.nuisance, estimands=("msm",)
        ),
    }


class TestALegacyResultKeepsItsNumbersAndRefusesARecomputation:
    """RM13: a restored result holds what it computed and computes nothing new.

    Loading checks nothing, so the stored estimates answer as they were saved. Every sweep
    recomputes through ``_retarget_detailed``, which checks the declaration as the fit does.
    """

    @pytest.fixture(scope="class")
    def result(self) -> Any:
        return fitted_known()

    def test_the_stored_interval_answers_unchanged(self, result: Any) -> None:
        old = legacy_result(result)
        for name, estimate in result.estimates.items():
            assert old[name].ci == estimate.ci
            assert old[name].psi == estimate.psi

    @pytest.mark.parametrize("entry", ["truncation_curve", "retarget"])
    def test_every_recomputation_refuses(self, result: Any, entry: str) -> None:
        old = legacy_result(result)
        assert_refused(recomputations(old)[entry], CapabilityError, UNDECLARED)

    @pytest.mark.parametrize("entry", ["truncation_curve", "retarget"])
    def test_the_declared_result_recomputes(self, result: Any, entry: str) -> None:
        """The control: the same entry on the result before the declaration was lost."""
        recomputations(result)[entry]()

    @pytest.mark.parametrize("entry", ["truncation_curve", "retarget"])
    def test_removing_the_retarget_check_fails_the_refusal(
        self, result: Any, entry: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        old = legacy_result(result)
        monkeypatch.setattr(tmle_module, "refuse_projection_weights", lambda model: None)
        with pytest.raises(AssertionError):
            assert_refused(recomputations(old)[entry], CapabilityError, UNDECLARED)


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
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(msm_module, "refuse_projection_weights", lambda model: None)
        for witness in declaration_witnesses():
            with pytest.raises(AssertionError):
                witness()

    @pytest.mark.parametrize("name", list(RESTORED))
    def test_removing_the_fit_layer_check_fails_the_fit_witnesses(
        self, name: str, monkeypatch
    ) -> None:
        kind, fragments = RESTORED[name]
        monkeypatch.setattr(tmle_module, "refuse_projection_weights", lambda model: None)
        monkeypatch.setattr(ltmle_module, "refuse_projection_weights", lambda model: None)
        with pytest.raises(AssertionError):
            assert_tmle_refuses(kind, *fragments)
        assert NeverFit.calls > 0, "the mutated TMLE fit refused before a learner"
        with pytest.raises(AssertionError):
            assert_ltmle_refuses(kind, *fragments)
        assert NeverFit.calls > 0, "the mutated LTMLE fit refused before a learner"

    def test_the_fit_layer_calls_the_shared_refusal(self) -> None:
        """One refusal, one text: the fit layer calls the function the declaration does."""
        assert tmle_module.refuse_projection_weights is refuse_projection_weights
        assert ltmle_module.refuse_projection_weights is refuse_projection_weights


# ------------------------------------------------------------------ the witness

#: A law on which a share weight is not a constant of the design.  ``g = 0.5`` in every
#: stratum makes the arm share 0.5, and ``Qbar`` differs across ``W`` so that the ``W``
#: coefficient reads the arm weights.  Every cell is a multiple of ``1 / N``.
SHARE_COUNTS = law.cell_counts(
    p_w=[0.4, 0.2, 0.4], g=[0.5, 0.5, 0.5], q=[[0.05, 0.05], [0.3, 0.7], [0.05, 0.95]]
)
SHARE_PROBS = SHARE_COUNTS / law.N


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


def beta_gateaux(weights: Any, point: int, *, step: float = 1e-30) -> np.ndarray:
    """The Gateaux derivative of ``law.msm_beta(P, weights)`` at one support point.

    The contamination path and the complex step of :func:`law.gateaux`.  When ``weights``
    is :func:`arm_share`, ``h`` is recomputed from the perturbed law, so the derivative
    carries the term through ``h`` that the fixed-weight derivative does not.
    """
    base = SHARE_PROBS.astype(complex)
    mass = np.zeros_like(base)
    mass[law.SUPPORT[point]] = 1.0
    perturbed = (1.0 - 1j * step) * base + 1j * step * mass
    return np.asarray(np.imag(law.msm_beta(perturbed, weights)) / step)


def eif(weights: Any) -> np.ndarray:
    """``(12, 3)``: the Gateaux derivative at every support point, for every term."""
    return np.array([beta_gateaux(weights, point) for point in range(len(law.SUPPORT))])


#: The reported SE over the exact estimated-share SE for ``msm[W]`` must stay below this.
#: The measured ratio is 0.7417.  A reported curve that carried the share term would give
#: 1.  The bound sits between the two, far enough from 0.7417 that it does not pin a
#: digit, and far enough from 1 that an understatement of 20 percent or more is the claim.
UNDERSTATEMENT_BOUND = 0.8


@pytest.fixture(scope="module")
def share_fit() -> Any:
    frame = law.frame(SHARE_COUNTS)
    dgp = law.DiscreteLaw(SHARE_PROBS)
    estimator = TMLE(
        outcome_learner=OracleOutcome(dgp),
        treatment_learner=OracleTreatment(dgp),
        cross_fit=False,
        msm=linear(weights=SampleShare(frame["A"]), weights_kind="known"),
        estimands="all",
        simultaneous=False,
        random_state=0,
    )
    return estimator.fit(frame, outcome="Y", treatment="A").single()


def first_rows() -> np.ndarray:
    counts = np.array([SHARE_COUNTS[cell] for cell in law.SUPPORT])
    return np.concatenate([[0], np.cumsum(counts)[:-1]])


def se_ratio(fit: Any, index: int) -> float:
    """The reported curve's SE over the exact estimated-share EIF's SE.

    Both are asymptotic: the reported curve's second moment on the exact sample, and the
    estimated-share EIF's variance under the law.  ``std_error`` itself divides by
    ``n - 1``, which would move this by a factor of 1.0005.
    """
    curve = np.asarray(fit.estimates[f"msm[{law.MSM_TERMS[index]}]"].influence_curve)
    probs = np.array([SHARE_PROBS[cell] for cell in law.SUPPORT])
    exact = float(probs @ eif(arm_share)[:, index] ** 2)
    return float(np.sqrt(np.mean(curve**2) / exact))


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
        curve = np.asarray(share_fit.estimates[f"msm[{law.MSM_TERMS[index]}]"].influence_curve)
        np.testing.assert_allclose(
            curve[first_rows()], eif(FROZEN_SHARE)[:, index], atol=1e-10, rtol=0
        )

    def test_the_oracle_carries_the_share_term(self) -> None:
        """The two oracles differ by exactly ``dbeta/dpi * (1{A = 1} - pi)``.

        ``dbeta/dpi`` comes from a complex step in the share alone, a path independent
        of the contamination path, so this checks that :func:`beta_gateaux` differentiates
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
        ratio = se_ratio(share_fit, law.MSM_TERMS.index("W"))
        assert ratio == pytest.approx(0.7417, abs=1e-4)
        assert ratio < UNDERSTATEMENT_BOUND

    def test_the_intercept_understates_its_standard_error(self, share_fit) -> None:
        ratio = se_ratio(share_fit, law.MSM_TERMS.index("(intercept)"))
        assert ratio == pytest.approx(0.9126, abs=1e-4)

    def test_control_the_arm_coefficient_is_unaffected(self, share_fit) -> None:
        """Control 2: on this design ``dbeta_a/dpi = 0``, so the term vanishes for ``a``."""
        assert se_ratio(share_fit, law.MSM_TERMS.index("a")) == pytest.approx(1.0, abs=1e-9)
