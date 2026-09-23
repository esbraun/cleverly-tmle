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
* every fit entry refuses a restored or modified regime before any learner or density call;
* a regime pickled before the field existed loads undeclared;
* a result restored with such a regime keeps its stored estimates, and every recomputation
  from it refuses;
* the simulated-confounding replay admits a frozen regime and refuses a restored one before
  its density runs;
* a deliberate mutation that removes a refusal makes those witnesses fail;
* on an exact law, the odds tilt of the sample mechanism declared ``"known"`` gets the
  fixed-density curve, which understates variance for the population-law odds-tilt target;
* a density that is known keeps its interval.

The exact-law gap is the reason for the refusal, measured without sampling error, as
``tests/unit/test_msm_projection_weights.py`` measures it for RM13.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

import cleverly._declarations as declarations_module
import cleverly.interventions.base as base_module
import tests.unit.test_msm_projection_weights as rm13_tests
from cleverly import CausalStudy, PointTreatment, RegimeMean
from cleverly.data import CausalData
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError, DataError
from cleverly.interventions import Incremental, Static, Stochastic
from cleverly.interventions.base import (
    _ESTIMATED_DENSITY,
    _UNDECLARED_DENSITY,
    refuse_regime_densities,
)
from cleverly.sensitivity import _simulated_confounding_fixed as replay_module
from cleverly.sensitivity import simulated_confounding
from tests import discrete_law as law
from tests.conftest import OracleOutcome, OracleTreatment, linear_in_sample
from tests.pickles import legacy_without
from tests.unit._confounding_support import Counter, forbid_draw_and_refit
from tests.unit._declaration_support import (
    PATHWISE,
    assert_every_witness_fails,
    assert_refused,
    assert_refused_before_any_call,
    recomputations,
    restored,
    restored_states,
    se_ratio,
    tmle_module,
)
from tests.unit._declaration_support import legacy_result as legacy_result_of
from tests.unit._natural_course_support import NeverFit, never_fit_learners
from tests.unit.test_msm_projection_weights import FixedWeight, linear
from tests.unit.test_simulated_confounding_policies import (
    _GRID,
    _alias,
    _estimate,
    _fit_policy,
    _stochastic_density,
    _study,
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


class SpyDensity(FixedDensity):
    """A known density that counts its calls.  A refusal must come before the first one."""

    calls: ClassVar[int] = 0

    def __call__(self, frame: Any) -> np.ndarray:
        type(self).calls += 1
        return super().__call__(frame)


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
    SpyDensity.calls = 0
    return coin(SpyDensity(), density_kind="known")


def tmle_fit(regime: Any, learners: dict[str, Any]) -> Any:
    estimator = TMLE(interventions=(regime,), cross_fit=False, simultaneous=False, **learners)
    return estimator.fit(law.frame(), outcome="Y", treatment="A")


def study_fit(regime: Any, learners: dict[str, Any]) -> Any:
    study = CausalStudy(
        law.frame(), design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",))
    )
    return study.identify(RegimeMean((regime,))).estimate(
        outcome_learner=learners["outcome_learner"],
        treatment_learner=learners["treatment_learner"],
        cross_fit=False,
        simultaneous=False,
    )


def direct_refit(regime: Any, learners: dict[str, Any]) -> Any:
    """``TMLE.refit`` on prepared data: the entry the refutations and the replay use."""
    estimator = TMLE(interventions=(regime,), cross_fit=False, simultaneous=False, **learners)
    return estimator.refit(causal_data())


#: Every fit entry that can reach a restored regime.
ENTRIES: dict[str, Callable[[Any, dict[str, Any]], Any]] = {
    "fit": tmle_fit,
    "study": study_fit,
    "refit": direct_refit,
}

#: What a restored regime can carry, and the refusal each one meets.
RESTORED = restored_states(UNDECLARED, ESTIMATED)


def assert_entry_refuses(entry: str, regime: Any, *fragments: str) -> None:
    assert_refused_before_any_call(
        lambda: ENTRIES[entry](regime, never_fit_learners()), SpyDensity, "density", *fragments
    )


class TestTheFitRefusesARestoredRegime:
    @pytest.mark.parametrize("entry", list(ENTRIES))
    @pytest.mark.parametrize("name", list(RESTORED))
    def test_every_entry_refuses_before_any_learner_or_density_call(
        self, entry: str, name: str
    ) -> None:
        kind, fragments = RESTORED[name]
        assert_entry_refuses(entry, restored(spy_coin(), "density_kind", kind), *fragments)

    def test_a_subclass_that_skips_the_declaration_refuses_at_the_fit(self) -> None:
        """The fit selects regimes with ``isinstance``, so a subclass cannot opt out."""

        class Skips(Stochastic):
            def __post_init__(self) -> None:
                pass

        SpyDensity.calls = 0
        regime = Skips(SpyDensity(), "coin")
        assert regime.density_kind is None
        assert_entry_refuses("fit", regime, UNDECLARED)

    @pytest.mark.parametrize("entry", list(ENTRIES))
    def test_a_known_declaration_reaches_the_first_learner(self, entry: str) -> None:
        """The control: the same entries with the declaration intact go past the check."""
        regime = spy_coin()
        with pytest.raises(AssertionError, match="before any learner is fitted"):
            ENTRIES[entry](regime, never_fit_learners())
        assert NeverFit.calls == 1
        assert SpyDensity.calls == 0, "a density ran before the first learner"


# ------------------------------------------------------------------ old pickles


def legacy(regime: Stochastic) -> Stochastic:
    """``regime`` as a pickle written before ``density_kind`` existed would restore it."""
    return legacy_without(regime, "density_kind")


class TestALegacyRegimeLoads:
    def test_a_pickle_without_the_field_reads_none_and_can_be_replaced(self) -> None:
        old = legacy(coin(density_kind="known"))
        assert "density_kind" not in vars(old)
        assert old.density_kind is None
        assert_refused(lambda: replace(old), CapabilityError, UNDECLARED)
        assert replace(old, density_kind="known").density_kind == "known"

    def test_a_legacy_regime_refuses_at_the_fit(self) -> None:
        SpyDensity.calls = 0
        assert_entry_refuses("fit", legacy(coin(SpyDensity(), density_kind="known")), UNDECLARED)


def stochastic_regimes(estimator: Any) -> list[Any]:
    return [item for item in estimator.interventions if isinstance(item, Stochastic)]


def legacy_result(result: Any) -> Any:
    """``result`` as an artifact written before ``density_kind`` existed would restore it."""
    return legacy_result_of(result, "density_kind", stochastic_regimes)


#: The estimands a retarget of the legacy result requests.
RETARGETED = ("ey_regime", "ate_regime")


class TestALegacyResultKeepsItsNumbersAndRefusesARecomputation:
    """RM25: a restored result holds what it computed and computes nothing new.

    Loading checks nothing, so the stored estimates answer as they were saved.  Every sweep
    recomputes through ``_retarget_detailed``, and a refit through
    ``_resolve_estimands_for_data``, and both check the declaration as the fit does.
    """

    @pytest.fixture(scope="class")
    def result(self) -> Any:
        regimes = (Static(0, name="never"), coin(density_kind="known"))
        estimator = TMLE(interventions=regimes, **linear_in_sample())
        return estimator.fit(law.frame(), outcome="Y", treatment="A").single()

    def test_the_stored_interval_answers_unchanged(self, result: Any) -> None:
        old = legacy_result(result)
        assert "ey_regime[coin]" in result.estimates
        for name, estimate in result.estimates.items():
            assert old[name].ci == estimate.ci
            assert old[name].psi == estimate.psi

    @pytest.mark.parametrize("entry", ["truncation_curve", "retarget", "refit"])
    def test_every_recomputation_refuses(self, result: Any, entry: str) -> None:
        old = legacy_result(result)
        assert_refused(recomputations(old, RETARGETED)[entry], CapabilityError, UNDECLARED)

    @pytest.mark.parametrize("entry", ["truncation_curve", "retarget", "refit"])
    def test_the_declared_result_recomputes(self, result: Any, entry: str) -> None:
        """The control: the same entry on the result before the declaration was lost."""
        recomputations(result, RETARGETED)[entry]()

    @pytest.mark.parametrize("entry", ["truncation_curve", "retarget", "refit"])
    def test_removing_the_fit_layer_check_fails_the_refusal(
        self, result: Any, entry: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        old = legacy_result(result)
        monkeypatch.setattr(tmle_module, "refuse_regime_densities", lambda interventions: None)
        with pytest.raises(AssertionError):
            assert_refused(recomputations(old, RETARGETED)[entry], CapabilityError, UNDECLARED)


# ------------------------------------------------------------------ the replay


def counted_fit() -> tuple[Any, Counter]:
    """A fitted regime mean whose density counts its calls, restored without its declaration.

    The counter is read off the restored result, because a pickle round trip copies it.
    """
    regime = Stochastic(Counter(_stochastic_density), name="policy", density_kind="known")
    old = legacy_result(_estimate(_study(), RegimeMean((regime,))))
    density = old.estimator.interventions[0].density_fn
    density.calls = 0
    return old, density


def validate(result: Any) -> Any:
    alias = _alias(result)
    return replay_module.validate_fixed_replay(result, alias, result.parameter_keys[alias])


class TestTheReplay:
    def test_a_known_regime_replays_through_frozen_regimes(self) -> None:
        result = _fit_policy()
        replay = validate(result)
        assert replay.interventions
        assert all(type(item) is replay_module._FrozenRegime for item in replay.interventions)
        assert refuse_regime_densities(replay.interventions) is None
        surface = simulated_confounding(
            result, estimand=_alias(result), grid=_GRID, random_state=31
        )
        assert all(cell.failure is None for cell in surface.cells)

    def test_a_legacy_regime_refuses_at_replay_before_its_density_runs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        old, density = counted_fit()
        forbid_draw_and_refit(monkeypatch, old.estimator)
        assert_refused(
            lambda: simulated_confounding(old, estimand=_alias(old), grid=_GRID, random_state=31),
            CapabilityError,
            UNDECLARED,
        )
        assert_refused(lambda: validate(old), CapabilityError, UNDECLARED)
        assert density.calls == 0, "the density was evaluated before the refusal"

    def test_removing_the_replay_check_evaluates_the_density(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The deliberate-mutation control: without the check the replay runs the density."""
        old, density = counted_fit()
        monkeypatch.setattr(replay_module, "refuse_regime_densities", lambda interventions: None)
        replay = validate(old)
        assert all(type(item) is replay_module._FrozenRegime for item in replay.interventions)
        assert density.calls == 1


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
    @pytest.mark.parametrize("name", list(RESTORED))
    def test_removing_the_fit_layer_check_fails_the_fit_witnesses(
        self, entry: str, name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        kind, fragments = RESTORED[name]
        monkeypatch.setattr(tmle_module, "refuse_regime_densities", lambda interventions: None)
        with pytest.raises(AssertionError):
            assert_entry_refuses(entry, restored(spy_coin(), "density_kind", kind), *fragments)
        assert NeverFit.calls > 0, "the mutated fit refused before a learner"

    def test_every_site_calls_the_one_refusal(self) -> None:
        """One refusal, one text: the fit layer and the replay call the declaration's check."""
        assert tmle_module.refuse_regime_densities is refuse_regime_densities
        assert replay_module.refuse_regime_densities is refuse_regime_densities
        assert base_module.refuse_regime_densities is refuse_regime_densities

    def test_removing_the_shared_declaration_fails_both_of_its_users(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RM13 and RM25 refuse through one ``FunctionDeclaration.refuse``.

        Each witness that the shared refusal decides must fail under the mutation.  The RM13
        unknown-value witness is not among them: ``refuse_projection_weights`` calls
        ``check`` itself, so that witness still refuses.
        """
        monkeypatch.setattr(
            declarations_module.FunctionDeclaration, "refuse", lambda self, kind: None
        )
        weight_suite = rm13_tests.TestTheDeclarationIsRequired()
        density_suite = TestTheDeclarationIsRequired()
        assert_every_witness_fails(
            [
                weight_suite.test_an_undeclared_callable_is_refused,
                weight_suite.test_an_estimated_weight_is_refused_by_its_missing_term,
                density_suite.test_an_undeclared_density_is_refused,
                density_suite.test_an_estimated_density_is_refused_by_its_missing_term,
                lambda: density_suite.test_an_unknown_declaration_is_refused("Known"),
            ]
        )
        assert linear(weights=FixedWeight()).weights_kind is None
        assert coin().density_kind is None


# ------------------------------------------------------------------ the shared check

#: Each user of the shared declaration: a build from one declaration value, the fragment of
#: its unknown-value refusal, and its field.  The uniform MSM has no weight, so its own check
#: of ``"estimated"`` without a weight must not read a value that is not a string first.
DECLARATION_USERS: dict[str, tuple[Callable[[Any], Any], str, str]] = {
    "msm weight": (
        lambda kind: linear(weights=FixedWeight(), weights_kind=kind),
        "weights_kind must be 'known', 'estimated' or None",
        "weights_kind",
    ),
    "uniform msm": (
        lambda kind: linear(weights_kind=kind),
        "weights_kind must be 'known', 'estimated' or None",
        "weights_kind",
    ),
    "regime density": (lambda kind: coin(density_kind=kind), UNKNOWN, "density_kind"),
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

#: A variant law on which the omitted term is material and asymmetric in delta.  ``g``
#: differs across ``W``, and ``Qbar(1, w) - Qbar(0, w)`` is 0.8 or 0.9 at every ``w``.
#: Every cell is a multiple of ``1 / N``, so the 1000 rows realise the law exactly.
TILT_COUNTS = law.cell_counts(
    p_w=[0.4, 0.4, 0.2], g=[0.25, 0.5, 0.75], q=[[0.05, 0.95], [0.1, 0.9], [0.1, 0.9]]
)
TILT_PROBS = TILT_COUNTS / law.N
CELL_P = np.array([TILT_PROBS[cell] for cell in law.SUPPORT])
W_OF = np.array([w for w, _, _ in law.SUPPORT])
A_OF = np.array([a for _, a, _ in law.SUPPORT], dtype=float)
DGP = law.DiscreteLaw(TILT_PROBS)

#: The two tilts, by the labels the oracle's ``ey_ipsi`` keys on.
DELTAS = {label: law.IPSI_DELTAS[label] for label in ("odds x2", "odds x0.5")}

#: The reported SE over the exact estimated-density SE at delta = 2 must stay below this.
#: The probe measured 0.6226 before this bound was chosen.  A curve that carried the
#: Kennedy term would read 1.  The bound claims an understatement of more than 30 percent and
#: pins no digit; the approx line records the measured value, as RM13 records 0.7417.
UNDERSTATEMENT_BOUND = 0.7


def levels(frame: Any) -> np.ndarray:
    return np.rint(np.asarray(frame["W"], dtype=float)).astype(int)


def odds_tilt(g: np.ndarray, delta: float) -> np.ndarray:
    """Kennedy's tilt, ``q(1 | w) = delta g / (delta g + 1 - g)``, as ``(3, 2)`` columns."""
    one = delta * g / (delta * g + 1.0 - g)
    return np.column_stack([1.0 - one, one])


class KnownTilt:
    """The tilt of the law's true mechanism, a fixed function of ``W``."""

    def __init__(self, delta: float) -> None:
        self.star = odds_tilt(DGP.g, delta)

    def __call__(self, frame: Any) -> np.ndarray:
        return self.star[levels(frame)]


class SampleTilt:
    """The user lie: the tilt of the sample's treated share in each stratum."""

    def __init__(self, sample: Any, delta: float) -> None:
        w, a = levels(sample), np.asarray(sample["A"], dtype=float)
        self.star = odds_tilt(np.array([a[w == k].mean() for k in range(3)]), delta)

    def __call__(self, frame: Any) -> np.ndarray:
        return self.star[levels(frame)]


def fixed_eif(star: np.ndarray, *, step: float = 1e-30) -> np.ndarray:
    """The Gateaux derivative of ``sum_w P(w) sum_a q*(a | w) Qbar(a, w)``, ``q*`` frozen.

    The contamination path and the complex step of :func:`law.gateaux`.  Only ``P(W)`` and
    ``Qbar`` move with the law; the density does not.
    """

    def psi(p: Any) -> Any:
        q = p[:, :, 1] / p.sum(axis=2)
        return (p.sum(axis=(1, 2)) * (star * q).sum(axis=1)).sum()

    out = []
    for point in range(len(law.SUPPORT)):
        base = TILT_PROBS.astype(complex)
        mass = np.zeros_like(base)
        mass[law.SUPPORT[point]] = 1.0
        out.append(float(np.imag(psi((1 - 1j * step) * base + 1j * step * mass)) / step))
    return np.array(out)


def kennedy_term(delta: float) -> np.ndarray:
    """``T = delta (Qbar(1, W) - Qbar(0, W)) / D^2 (A - g(W))``, at every support point."""
    g, q = DGP.g, DGP.q
    slope = delta * (q[:, 1] - q[:, 0]) / (delta * g + 1.0 - g) ** 2
    return slope[W_OF] * (A_OF - g[W_OF])


def oracle_tmle(**axis: Any) -> Any:
    """In sample, with the law's own nuisances, so the curve has no learner error."""
    estimator = TMLE(
        outcome_learner=OracleOutcome(DGP),
        treatment_learner=OracleTreatment(DGP),
        cross_fit=False,
        simultaneous=False,
        random_state=0,
        **axis,
    )
    return estimator.fit(law.frame(TILT_COUNTS), outcome="Y", treatment="A").single()


def tilt_curve(result: Any) -> np.ndarray:
    return np.asarray(result.estimates["ey_regime[tilt]"].influence_curve)


@pytest.fixture(scope="module")
def sample_fits() -> dict[str, Any]:
    """The sample tilt, declared ``"known"``: the lie that no declaration can detect."""
    sample = law.frame(TILT_COUNTS)
    return {
        label: oracle_tmle(
            interventions=(Stochastic(SampleTilt(sample, delta), "tilt", density_kind="known"),)
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
        ipsi = oracle_tmle(incremental=(Incremental(2.0, name="odds x2"),))
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
        result = oracle_tmle(
            interventions=(Stochastic(KnownTilt(2.0), "tilt", density_kind="known"),)
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
