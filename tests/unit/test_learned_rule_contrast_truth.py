"""The truth helpers of a learned-rule fit beside a known static regime (X11 (c)).

:mod:`tests.studies._learned_rule_contrast_law` gives the fixed value of ``Static(a)`` on the
RM30 quadrature and the truth of the learned-rule contrast.  Each is checked against a closed
form where one exists and against Monte Carlo draws.  The fits draw from throwaway seeds at a
small size, and no test asserts a bias or a coverage.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
import pytest
from scipy.special import expit

from cleverly.interventions import Static
from tests.studies import _learned_rule_contrast_law as contrast
from tests.studies import _learned_rule_law as base

#: Throwaway seeds, far from every declared and pilot seed.
THROWAWAY = 9_970_000
#: Monte Carlo draws of W, and the multiple of the Monte Carlo SE a check allows.
DRAWS = 400_000
SE_MULTIPLE = 4.0


def _closed_form(arm: float) -> float:
    """``E expit(c + s W1)`` on the ``non_exceptional`` law, from the logistic integral."""
    total = 0.0
    for w2 in base.W2_LEVELS:
        intercept = 0.2 - 0.3 * w2 + 0.1 * arm
        slope = 0.5 + arm
        integral = (np.logaddexp(0.0, intercept + slope) - np.logaddexp(0.0, intercept - slope)) / (
            slope
        )
        total += 0.5 * integral / 2.0
    return float(total)


def _covariates(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, DRAWS), rng.binomial(1, 0.5, DRAWS).astype(float)


@cache
def _fit() -> Any:
    frame = base.draw("non_exceptional", 400, THROWAWAY)
    return contrast.fit(frame, base.outcome_learner(), THROWAWAY + 1, arms=(1, 0))


class TestTheStaticValue:
    @pytest.mark.parametrize("arm", [0.0, 1.0])
    def test_it_is_the_closed_form_on_the_non_exceptional_law(self, arm: float) -> None:
        assert contrast.static_value(arm, "non_exceptional") == pytest.approx(
            _closed_form(arm), abs=1e-8
        )

    @pytest.mark.parametrize("law", ["non_exceptional", "misspecified_limit"])
    @pytest.mark.parametrize("arm", [0.0, 1.0])
    def test_it_matches_monte_carlo_draws_of_the_outcome(self, law: str, arm: float) -> None:
        """Draw ``Y^a`` itself, so the check reads the law and not only the quadrature."""
        w1, w2 = _covariates(THROWAWAY + 10 + int(arm))
        y = np.random.default_rng(THROWAWAY + 20).binomial(1, base.qbar0(arm, w1, w2, law))
        se = float(np.std(y, ddof=1)) / np.sqrt(DRAWS)
        assert abs(float(np.mean(y)) - contrast.static_value(arm, law)) < SE_MULTIPLE * se

    def test_the_arms_differ_so_the_two_contrasts_differ(self) -> None:
        """Nonzero witness: the two static values, and so the two contrast truths, differ."""
        gap = contrast.static_value(1.0, "non_exceptional") - contrast.static_value(
            0.0, "non_exceptional"
        )
        assert gap > 0.01

    def test_an_arm_of_no_static_regime_refuses(self) -> None:
        with pytest.raises(ValueError, match="arm 0 or 1"):
            contrast.static_value(0.5, "non_exceptional")


class TestTheContrastTruth:
    def test_the_fit_runs_with_the_constructor_s_defaults(self) -> None:
        result = _fit()
        assert result.estimator.simultaneous
        assert result.simultaneous is None
        assert "ate_learned_rule[learned rule vs always 0]" in result.estimates

    @pytest.mark.parametrize(("arm", "column"), [(1.0, 1), (0.0, 2)])
    def test_it_is_the_harness_truth_minus_the_static_value(self, arm: float, column: int) -> None:
        result = _fit()
        learner = base.outcome_learner()
        truth = contrast.contrast_truth(result, learner, "non_exceptional", arm, column=column)
        assert truth.learned == base.harness(result, base.outcome_learner(), "non_exceptional")
        assert truth.known == contrast.static_value(arm, "non_exceptional")
        assert truth.contrast == truth.learned.truth - truth.known
        flipped = contrast.contrast_truth(
            result, learner, "non_exceptional", arm, column=column, reference_is_rule=True
        )
        assert flipped.contrast == -truth.contrast

    @pytest.mark.parametrize(("arm", "column"), [(1.0, 1), (0.0, 2)])
    def test_it_matches_monte_carlo_draws_of_the_fold_rules(self, arm: float, column: int) -> None:
        """Average ``Qbar0(d_v(W), W) - Qbar0(a, W)`` over the refitted fold rules and draws."""
        result = _fit()
        w1, w2 = _covariates(THROWAWAY + 30)
        points = np.column_stack([w1, w2])
        differences = np.zeros(DRAWS)
        folds = len(list(result.nuisance.folds))
        for fold in range(folds):
            rule = base.fold_rule(result, base.outcome_learner(), fold)
            treat = np.asarray(rule(points), dtype=float)
            differences += base.qbar0(treat, w1, w2, "non_exceptional") / folds
        differences -= base.qbar0(arm, w1, w2, "non_exceptional")
        se = float(np.std(differences, ddof=1)) / np.sqrt(DRAWS)
        truth = contrast.contrast_truth(
            result, base.outcome_learner(), "non_exceptional", arm, column=column
        )
        assert abs(float(np.mean(differences)) - truth.contrast) < SE_MULTIPLE * se

    def test_control_a_column_of_another_regime_refuses(self) -> None:
        """Mutation: ``Static(1)``'s truth read against the ``Static(0)`` column refuses."""
        with pytest.raises(ValueError, match="not Static"):
            contrast.contrast_truth(
                _fit(), base.outcome_learner(), "non_exceptional", 1.0, column=2
            )

    def test_the_estimator_declares_each_arm(self) -> None:
        declared = contrast.estimator(base.outcome_learner(), 0, arms=(1, 0))
        assert [regime.name for regime in declared.interventions] == [
            Static(1).name,
            Static(0).name,
        ]
        assert declared.n_folds == base.N_FOLDS


def test_the_quadrature_is_the_expit_of_the_declared_logit() -> None:
    """Premise: the law's ``Qbar0`` is the declared logistic form the closed form integrates."""
    w1 = np.linspace(-1.0, 1.0, 7)
    np.testing.assert_allclose(
        base.qbar0(1.0, w1, 1.0, "non_exceptional"),
        expit(0.2 + 0.5 * w1 - 0.3 + (0.1 + w1)),
        rtol=0,
        atol=1e-15,
    )
