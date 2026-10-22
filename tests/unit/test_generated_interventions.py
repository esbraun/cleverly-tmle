"""Generated-outcome truths for regimes, incremental tilts and MSM coefficients (X82 part E3).

Each truth is checked against a table worked by hand or against a closed form. Each mask,
weight and law the truth reads has a mutation control that moves the truth when it is
dropped. The end-to-end tests refit a real TMLE 400 times on each law.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import (
    CausalStudy,
    IncrementalEffect,
    MSMProjection,
    PointTreatment,
    RegimeContrast,
)
from cleverly.exceptions import CapabilityError
from cleverly.interventions import Incremental, Rule, Static
from cleverly.interventions.incremental import _tilt_general
from cleverly.msm import MSM, solve_projection
from cleverly.validation import GaussianAdjustmentOutcome, refute
from cleverly.validation._point_generated import InterventionLaw, PointTarget, Projection

ARMS = (0.0, 1.0)


def _target(kind: str, **laws: Any) -> PointTarget:
    return PointTarget(
        kind=kind,  # type: ignore[arg-type]
        population="all",
        value_code=None,
        reference_code=None,
        arms=laws.pop("arms", ARMS),
        **laws,
    )


def _expit(values: Any) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-np.asarray(values, dtype=float))))


MEANS = {0.0: np.array([1.0, 2.0, 3.0]), 1.0: np.array([2.0, 5.0, 4.0])}
RULE_LAW = InterventionLaw("rule", values=np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]))
NEVER = InterventionLaw("never", values=np.array([[1.0, 0.0]] * 3))


class TestRegimeTruth:
    def test_a_regime_mean_by_hand(self) -> None:
        # Row values 2, 2 and 3.5 under the rule: their mean is 2.5.
        target = _target("intervention", value_law=RULE_LAW)
        assert target.truth(MEANS, np.ones(3)) == pytest.approx(2.5, abs=1e-15)
        # Mutation: reading the density columns in the reverse arm order.
        flipped = InterventionLaw("rule", values=RULE_LAW.values[:, ::-1])
        mutated = _target("intervention", value_law=flipped)
        assert mutated.truth(MEANS, np.ones(3)) != pytest.approx(2.5)

    def test_a_regime_contrast_and_its_weights(self) -> None:
        target = _target("intervention_difference", value_law=RULE_LAW, reference_law=NEVER)
        assert target.truth(MEANS, np.ones(3)) == pytest.approx(0.5, abs=1e-15)
        # Weights 1, 0, 1 keep rows 1 and 3: (2 + 3.5) / 2 - (1 + 3) / 2.
        weights = np.array([1.0, 0.0, 1.0])
        assert target.truth(MEANS, weights) == pytest.approx(0.75, abs=1e-15)

    def test_the_covariate_part_reads_the_contrast(self) -> None:
        # A Gaussian contrast of two static regimes is constant over the rows.
        always = InterventionLaw("always", values=np.array([[0.0, 1.0]] * 3))
        additive = {0.0: np.array([1.0, 2.0, 3.0]), 1.0: np.array([1.5, 2.5, 3.5])}
        static = _target("intervention_difference", value_law=always, reference_law=NEVER)
        assert static.covariate_part_vanishes(additive, np.ones(3)) is True
        # A rule that reads W leaves a covariate part, and so does one regime's level.
        rule = _target("intervention_difference", value_law=RULE_LAW, reference_law=NEVER)
        assert rule.covariate_part_vanishes(additive, np.ones(3)) is False
        level = _target("intervention", value_law=always)
        assert level.covariate_part_vanishes(additive, np.ones(3)) is False


class TestTiltTruth:
    def test_an_incremental_mean_on_two_rows_by_hand(self) -> None:
        # g(1 | W) = 0.2 and 0.6 and the odds factor 2 give q(1 | W) = 1/3 and 3/4.
        mechanism = {0.0: np.array([0.8, 0.4]), 1.0: np.array([0.2, 0.6])}
        means = {0.0: np.array([0.0, 1.0]), 1.0: np.array([1.0, 3.0])}
        tilt = InterventionLaw("odds x2", factors=np.array([1.0, 2.0]))
        target = _target("intervention", value_law=tilt)
        expected = (1.0 / 3.0 + (0.75 * 3.0 + 0.25 * 1.0)) / 2.0
        assert target.truth(means, np.ones(2), mechanism) == pytest.approx(expected, abs=1e-15)
        # Mutation: the observed g in place of the tilted one gives the natural course.
        untilted = _target("intervention", value_law=InterventionLaw("odds x1", factors=np.ones(2)))
        assert untilted.truth(means, np.ones(2), mechanism) == pytest.approx(1.2, abs=1e-15)
        assert expected - 1.2 > 0.2

    def test_a_tilt_needs_the_true_mechanism(self) -> None:
        tilt = InterventionLaw("odds x2", factors=np.array([1.0, 2.0]))
        means = {0.0: np.zeros(2), 1.0: np.ones(2)}
        with pytest.raises(ValueError, match="needs the true mechanism"):
            _target("intervention", value_law=tilt).truth(means, np.ones(2))

    def test_three_arms_agree_with_the_package_tilt(self) -> None:
        rng = np.random.default_rng(2)
        mechanism = rng.dirichlet(np.ones(3), size=50)
        factors = np.array([1.0, 3.0, 0.5])
        ours = InterventionLaw("tilt", factors=factors).density(mechanism)
        package = _tilt_general((tuple(factors),), mechanism)[0][:, :, 0]
        np.testing.assert_allclose(ours, package, rtol=0.0, atol=1e-15)


def _msm_arrays(n: int, *, seed: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A non-saturated design ``(1, a, W)``, weights that read ``W`` and ``a``, and ``W``."""
    covariate = np.random.default_rng(seed).normal(size=n)
    design = np.stack(
        [np.column_stack([np.ones(n), np.full(n, arm), covariate]) for arm in ARMS], axis=1
    )
    weights = np.column_stack([0.5 + (covariate > 0.0), 1.8 * (0.5 + (covariate > 0.0))])
    weights[:, 0] *= 0.6 / 1.8
    return design, weights, covariate


class TestProjection:
    def test_a_linear_msm_slope_is_the_effect(self) -> None:
        n, effect = 40, 0.7
        signal = np.random.default_rng(1).normal(size=n)
        means = {0.0: signal, 1.0: signal + effect}
        design = np.stack([np.column_stack([np.ones(n), np.full(n, arm)]) for arm in ARMS], axis=1)
        slope = _target("msm", projection=Projection(design, np.ones((n, 2)), "identity", term=1))
        assert slope.truth(means, np.ones(n)) == pytest.approx(effect, abs=1e-12)
        # The slope's covariate part is 0 at every row, and the intercept's is f(W) - mean f.
        assert slope.covariate_part_vanishes(means, np.ones(n)) is True
        intercept = Projection(design, np.ones((n, 2)), "identity", term=0)
        table = np.column_stack([means[0.0], means[1.0]])
        part = intercept.covariate_part(table, np.ones(n))
        np.testing.assert_allclose(part, signal - signal.mean(), atol=1e-12)
        target = _target("msm", projection=intercept)
        assert target.covariate_part_vanishes(means, np.ones(n)) is False

    @pytest.mark.parametrize("link", ["identity", "log", "logit"])
    def test_each_link_agrees_with_the_package_projection(self, link: str) -> None:
        n = 200
        design, weights, covariate = _msm_arrays(n)
        # A logistic law with an interaction, so no working model below is saturated.
        table = np.column_stack(
            [_expit(-0.5 + covariate**2), _expit(0.3 + covariate + 0.8 * covariate**2)]
        )
        population = np.random.default_rng(6).uniform(0.5, 1.5, size=n)
        ours = Projection(design, weights, link, term=1)
        package = solve_projection(design, weights, table, population, link)
        np.testing.assert_allclose(
            ours.coefficients(table, population), package.beta, rtol=0.0, atol=1e-9
        )
        # The covariate part premultiplies by the inverse of the same matrix M.
        fitted = design @ package.beta
        mean = {"identity": fitted, "log": np.exp(fitted), "logit": _expit(fitted)}[link]
        slope = {"identity": np.ones_like(mean), "log": mean, "logit": mean * (1 - mean)}[link]
        rows = np.einsum("ik,ikp->ip", weights * slope * (table - mean), design)
        expected = np.linalg.solve(package.jacobian, rows.T)[1]
        np.testing.assert_allclose(
            ours.covariate_part(table, population), expected, rtol=1e-7, atol=1e-10
        )

    def test_dropping_the_working_weights_moves_the_coefficient(self) -> None:
        n = 200
        design, weights, covariate = _msm_arrays(n)
        table = np.column_stack([_expit(covariate**2), _expit(1.0 + covariate + covariate**2)])
        weighted = Projection(design, weights, "identity", term=1)
        unweighted = Projection(design, np.ones_like(weights), "identity", term=1)
        moved = weighted.coefficient(table, np.ones(n)) - unweighted.coefficient(table, np.ones(n))
        assert abs(moved) > 1e-2


# ------------------------------------------------------------------ end to end


def _frame(n: int = 300, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    covariates = rng.normal(size=(n, 2))
    treatment = (rng.random(n) < _expit(0.5 * covariates[:, 0])).astype(int)
    return pd.DataFrame(
        {"W1": covariates[:, 0], "W2": covariates[:, 1], "A": treatment, "Y": rng.normal(size=n)}
    )


def _fit(estimand: Any, frame: pd.DataFrame | None = None) -> Any:
    study = CausalStudy(
        _frame() if frame is None else frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2")),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return study.identify(estimand).estimate(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            random_state=1,
            cross_fit=False,
        )


def _rule(frame: Any) -> np.ndarray:
    return (np.asarray(frame["W1"]) > 0.0).astype(int)


RULE = Rule(_rule, name="rule", rule_kind="known")
PROCESS = GaussianAdjustmentOutcome(effect=1.0, adjustment_scale=1.0)


def _run(result: Any, estimand: str, test: str, draws: int = 400) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return refute(
            result,
            estimand=estimand,
            tests=(test,),
            simulated_outcome=PROCESS,
            n_replicates=draws,
            random_state=2,
        )[test]


@pytest.fixture(scope="module")
def regimes() -> Any:
    return _fit(RegimeContrast((Static(0, name="never"), RULE), reference="never"))


@pytest.fixture(scope="module")
def tilts() -> Any:
    return _fit(IncrementalEffect((Incremental(1.0), Incremental(2.0))))


@pytest.fixture(scope="module")
def msm() -> Any:
    return _fit(MSMProjection(MSM.linear()))


class TestEndToEnd:
    def test_a_rule_contrast_in_the_held_design(self, regimes: Any) -> None:
        test = _run(regimes, "ate_regime[rule vs never]", "simulated_outcome")
        share = float(np.mean(_frame()["W1"] > 0.0))
        assert test.declared_effect == pytest.approx(share, abs=1e-12)
        # The rule reads W, so the contrast has a covariate part and reads bias only.
        assert test.passed is True and [r.name for r in test.readings] == ["bias"]

    def test_an_incremental_contrast_in_the_plasmode(self, tilts: Any) -> None:
        test = _run(tilts, "ate_ipsi[odds x2 vs natural course]", "plasmode")
        assert test.passed is True and len(test.readings) == 3

    def test_an_msm_slope_and_intercept_in_the_held_design(self, msm: Any) -> None:
        slope = _run(msm, "msm[a]", "simulated_outcome")
        assert slope.declared_effect == pytest.approx(1.0, abs=1e-12)
        assert slope.passed is True and len(slope.readings) == 3
        intercept = _run(msm, "msm[(intercept)]", "simulated_outcome", draws=10)
        assert [r.name for r in intercept.readings] == ["bias"]


class TestRefusals:
    def test_an_incremental_target_in_the_held_design(self, tilts: Any) -> None:
        with pytest.raises(CapabilityError, match="held design does not know") as caught:
            _run(tilts, "ate_ipsi[odds x2 vs natural course]", "simulated_outcome", draws=10)
        assert caught.value.refusal.reason == "api_redirection"

    def test_a_key_that_names_no_declared_regime(self, regimes: Any) -> None:
        alias = "ate_regime[rule vs never]"
        keys = dict(regimes.parameter_keys)
        keys[alias] = replace(keys[alias], value="other")
        tampered = replace(regimes, parameter_keys=keys)
        with pytest.raises(CapabilityError, match="do not name"):
            _run(tampered, alias, "simulated_outcome", draws=10)
