"""Generated-outcome truths for modified treatment policies (X82 part E3b).

A policy moves each unit's natural treatment, so its truth reads the natural treatment: the
held arm or dose of each row in the held design, and the frozen treatment model in the
plasmode. Each truth is checked by hand, and each law the truth reads has a mutation control.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CausalStudy, ModifiedTreatmentPolicyEffect, PointTreatment
from cleverly.exceptions import CapabilityError
from cleverly.interventions import DoseQuantile, ModifiedPolicy, Randomizer, Shift
from cleverly.validation import BernoulliAdjustmentOutcome, GaussianAdjustmentOutcome, refute
from cleverly.validation._point_generated import PlasmodeLaw, arm_means, held_mixture
from cleverly.validation.refute import _policy_laws, _validate_generated_eligibility

PROCESS = GaussianAdjustmentOutcome(effect=1.0, adjustment_scale=1.0)


def _expit(x: Any) -> Any:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _frame(n: int = 300, seed: int = 3, *, continuous: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    covariates = rng.normal(size=(n, 2))
    if continuous:
        treatment: Any = covariates[:, 0] + rng.normal(size=n)
    else:
        treatment = (rng.random(n) < _expit(0.5 * covariates[:, 0])).astype(int)
    return pd.DataFrame(
        {"W1": covariates[:, 0], "W2": covariates[:, 1], "A": treatment, "Y": rng.normal(size=n)}
    )


def _fit(policies: tuple[Any, ...], *, continuous: bool = False) -> Any:
    kind: dict[str, Any] = {"treatment_kind": "continuous"} if continuous else {}
    design = PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2"), **kind)
    study = CausalStudy(_frame(continuous=continuous), design=design)
    learners: dict[str, Any] = {
        "outcome_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
    }
    if continuous:
        # A Gaussian dose shifted by delta has a log density ratio linear in (a, W), so the
        # logistic classifier ratio is well specified.
        # An unpenalized fit keeps the ratio from shrinking toward 1.
        learners["ratio"] = "classifier"
        learners["treatment_learner"] = LogisticRegression(C=1e6, max_iter=1000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return study.identify(ModifiedTreatmentPolicyEffect(policies)).estimate(
            random_state=1, cross_fit=False, **learners
        )


def _natural(a: Any, h: Any) -> Any:
    return np.asarray(a)


def _treat_all(a: Any, h: Any) -> Any:
    return np.ones_like(np.asarray(a))


def _coin(a: Any, h: Any, e: Any) -> Any:
    return np.ones_like(np.asarray(a)) if e else np.asarray(a)


NATURAL = ModifiedPolicy("natural", apply=_natural, policy_kind="known")
TREAT_ALL = ModifiedPolicy("treat all", apply=_treat_all, policy_kind="known")
COIN = ModifiedPolicy(
    "coin", apply=_coin, randomizer=Randomizer((True, False), (0.3, 0.7)), policy_kind="known"
)


def _run(result: Any, estimand: str, test: str, draws: int = 400, process: Any = PROCESS) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return refute(
            result,
            estimand=estimand,
            tests=(test,),
            simulated_outcome=process,
            n_replicates=draws,
            random_state=2,
        )[test]


@pytest.fixture(scope="module")
def discrete() -> Any:
    return _fit((NATURAL, TREAT_ALL, COIN))


@pytest.fixture(scope="module")
def shifts() -> Any:
    return _fit((Shift(0.0, cap=50.0), Shift(0.5, cap=50.0)), continuous=True)


@pytest.fixture(scope="module")
def capped() -> Any:
    return _fit((Shift(0.0, cap=50.0), Shift(0.5, cap=0.0, name="+0.5 capped")), continuous=True)


def _laws(result: Any, alias: str) -> dict[str, Any]:
    key = result.parameter_keys[alias]
    return _policy_laws(result, result.identified_effect.functional, key, "simulated_outcome")


class TestDiscreteTruth:
    def test_the_transition_of_a_deterministic_and_a_randomized_policy(self, discrete: Any) -> None:
        laws = _laws(discrete, "ate_policy[coin vs natural]")
        coin, natural = laws["value_law"].transition, laws["reference_law"].transition
        # The natural course keeps each arm; the coin moves arm 0 to arm 1 with p = 0.3.
        assert np.array_equal(natural, np.broadcast_to(np.eye(2), natural.shape))
        expected = np.array([[0.7, 0.3], [0.0, 1.0]])
        assert np.allclose(coin, np.broadcast_to(expected, coin.shape), atol=1e-15)

    def test_the_held_truth_by_hand(self, discrete: Any) -> None:
        data = discrete.data
        held = np.asarray(_frame()["A"], dtype=float)
        process = BernoulliAdjustmentOutcome(intercept=-0.3, effect=1.2, adjustment_scale=1.5)
        for alias, share in (
            ("ate_policy[treat all vs natural]", 1.0),
            ("ate_policy[coin vs natural]", 0.3),
        ):
            target = _target(discrete, alias)
            means = arm_means(process, np.asarray(data.covariates), target.codes, data)
            weights = np.ones(data.n)
            mixture = held_mixture(target, data)
            # Only the untreated rows move, each to arm 1 with probability ``share``.
            gain = share * (1.0 - held) * (means[1.0] - means[0.0])
            assert target.truth(means, weights, mixture) == pytest.approx(float(np.mean(gain)))
            # Mutation: every row read as treated in place of its held arm moves the truth.
            treated = {0.0: np.zeros(data.n), 1.0: np.ones(data.n)}
            assert abs(target.truth(means, weights, treated) - np.mean(gain)) > 1e-2

    def test_the_plasmode_truth_reads_the_frozen_mechanism(self, discrete: Any) -> None:
        data = discrete.data
        target = _target(discrete, "ate_policy[treat all vs natural]")
        law = PlasmodeLaw.fit(discrete)
        means, weights, mixture = law.means(PROCESS, data, target)
        assert mixture is not None
        # Under a Gaussian process the move from arm 0 to arm 1 adds ``effect`` to the mean.
        expected = PROCESS.effect * float(np.mean(mixture[0.0]))
        assert target.truth(means, weights, mixture) == pytest.approx(expected, abs=1e-12)
        # Mutation: the arm columns of the frozen mechanism swapped move the truth.
        swapped = {0.0: mixture[1.0], 1.0: mixture[0.0]}
        assert abs(target.truth(means, weights, swapped) - expected) > 1e-2


def _target(result: Any, alias: str) -> Any:
    return _validate_generated_eligibility(result, alias, PROCESS, "simulated_outcome")


class TestContinuousTruth:
    def test_a_shift_under_a_linear_mean_is_effect_times_delta(self, shifts: Any) -> None:
        laws = _laws(shifts, "ate_policy[+0.5 vs natural course]")
        held = np.asarray(_frame(continuous=True)["A"], dtype=float)
        assert np.array_equal(laws["doses"][0.0], held + 0.5)
        assert np.array_equal(laws["doses"][1.0], held)
        test = _run(shifts, "ate_policy[+0.5 vs natural course]", "simulated_outcome")
        assert test.declared_effect == pytest.approx(0.5 * PROCESS.effect, abs=1e-12)
        # The contrast is the constant effect * delta at every row, so it reads all three.
        assert test.passed is True and len(test.readings) == 3

    def test_a_capped_shift_moves_only_the_rows_below_the_cap(self, capped: Any) -> None:
        alias = "ate_policy[+0.5 capped vs natural course]"
        held = np.asarray(_frame(continuous=True)["A"], dtype=float)
        moved = held + 0.5 <= 0.0
        test = _run(capped, alias, "simulated_outcome", draws=10)
        assert test.declared_effect == pytest.approx(0.5 * float(np.mean(moved)), abs=1e-12)
        # The cap makes the contrast vary over rows, so it reads bias only.
        assert [reading.name for reading in test.readings] == ["bias"]


class TestEndToEnd:
    def test_a_discrete_policy_in_the_held_design(self, discrete: Any) -> None:
        test = _run(discrete, "ate_policy[treat all vs natural]", "simulated_outcome")
        held = np.asarray(_frame()["A"], dtype=float)
        assert test.declared_effect == pytest.approx(float(np.mean(1.0 - held)), abs=1e-12)
        assert test.passed is True and [r.name for r in test.readings] == ["bias"]

    def test_a_randomized_policy_in_the_plasmode(self, discrete: Any) -> None:
        test = _run(discrete, "ate_policy[coin vs natural]", "plasmode")
        assert test.passed is True


class TestRefusals:
    def test_a_continuous_plasmode(self, shifts: Any) -> None:
        with pytest.raises(CapabilityError, match="no law of a continuous treatment"):
            _run(shifts, "ate_policy[+0.5 vs natural course]", "plasmode", draws=10)

    def test_arm_effects_on_a_continuous_treatment(self, shifts: Any) -> None:
        process = GaussianAdjustmentOutcome(arm_effects={0: 0.0, 1: 1.0})
        with pytest.raises(CapabilityError, match="arm_effects on a continuous treatment"):
            _run(shifts, "ate_policy[+0.5 vs natural course]", "simulated_outcome", 10, process)

    def test_a_cap_that_a_tuner_resolves(self) -> None:
        result = _fit(
            (Shift(0.0, cap=50.0), Shift(0.5, cap=DoseQuantile(0.9), target="population")),
            continuous=True,
        )
        with pytest.raises(CapabilityError, match="cannot evaluate the declared policy"):
            _run(result, "ate_policy[+0.5 vs natural course]", "simulated_outcome", draws=10)

    def test_a_held_policy_with_a_missing_treatment(self, discrete: Any) -> None:
        data = _MissingTreatment(discrete.data)
        with pytest.raises(CapabilityError, match="unrecorded row's treatment is unknown"):
            _run(
                replace(discrete, data=data),
                "ate_policy[treat all vs natural]",
                "simulated_outcome",
                10,
            )

    def test_a_key_that_names_no_declared_policy(self, discrete: Any) -> None:
        alias = "ate_policy[treat all vs natural]"
        keys = dict(discrete.parameter_keys)
        keys[alias] = replace(keys[alias], value="other")
        with pytest.raises(CapabilityError, match="do not name"):
            _run(replace(discrete, parameter_keys=keys), alias, "simulated_outcome", draws=10)


class _MissingTreatment:
    """Forward to held data, and report a missing treatment."""

    has_missing_treatment = True

    def __init__(self, data: Any) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        return getattr(self._data, name)
