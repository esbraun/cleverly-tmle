"""Regimes end to end through the estimator.

The claim that matters most is :meth:`TestAStaticFitIsTheArmFit.test_the_estimates_are
_identical`: with static regimes the regime path must reproduce the arm path *bit for
bit*, not merely to a tolerance. Everything the package says about the arm-indexed
estimands -- their oracle checks, their coverage studies, their fixtures -- then carries
over to the regime path unchanged, and only for as long as that equality holds.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly import load
from cleverly.datasets import make_linear_ate
from cleverly.estimators import TMLE
from cleverly.exceptions import DataError
from cleverly.interventions import Rule, Static, Stochastic
from tests.conftest import FAST_KWARGS


@pytest.fixture(scope="module")
def frame():
    return make_linear_ate(n=400, seed=7)[0]


def fit(frame, **overrides):
    return TMLE(**{**FAST_KWARGS, **overrides}).fit(frame, outcome="Y", treatment="A").single()


class TestAStaticFitIsTheArmFit:
    """The regression surface: a static regime must preserve the binary arm path."""

    @pytest.fixture(scope="class")
    def pair(self, frame):
        arms = fit(frame, estimands=("ey0", "ey1", "ate"))
        regimes = fit(frame, interventions=(Static(0), Static(1)))
        return arms, regimes

    def test_the_estimates_are_identical(self, pair) -> None:
        arms, regimes = pair
        for arm_name, regime_name in (
            ("ey0", "ey_regime[always 0]"),
            ("ey1", "ey_regime[always 1]"),
            ("ate", "ate_regime[always 1 vs always 0]"),
        ):
            assert regimes.estimates[regime_name].psi == arms.estimates[arm_name].psi
            assert np.array_equal(
                regimes.estimates[regime_name].influence_curve,
                arms.estimates[arm_name].influence_curve,
            )

    def test_the_clever_covariate_is_the_same_arithmetic(self, pair) -> None:
        """Not merely the same estimate: the same columns went into the fluctuation."""
        arms, regimes = pair
        mean = arms.fluctuations["mean"]
        regime = regimes.fluctuations["regime"]
        np.testing.assert_array_equal(np.sort(regime.epsilon), np.sort(mean.epsilon))

    def test_bare_levels_are_read_as_static_regimes(self, frame) -> None:
        explicit = fit(frame, interventions=(Static(0), Static(1)))
        bare = fit(frame, interventions=(0, 1))
        assert bare.estimates.keys() == explicit.estimates.keys()
        assert bare["ey_regime[always 1]"].psi == explicit["ey_regime[always 1]"].psi


class TestARuleIsEstimated:
    @pytest.fixture(scope="class")
    def result(self, frame):
        return fit(
            frame,
            interventions=(
                Static(0, name="never"),
                Rule(lambda w: (np.asarray(w["W1"]) > 0).astype(int), name="treat if W1 > 0"),
            ),
        )

    def test_it_reports_a_mean_per_regime_and_one_contrast(self, result) -> None:
        assert list(result.estimates) == [
            "ey_regime[never]",
            "ey_regime[treat if W1 > 0]",
            "ate_regime[treat if W1 > 0 vs never]",
        ]

    def test_two_regimes_keep_their_labels(self, result) -> None:
        """A two-arm fit collapses to ``ate``; two *regimes* must not.

        ``ate`` is kept short because it is historical and unambiguous. "The average
        effect" of one regime against another is neither.
        """
        assert "ate" not in result.estimates

    def test_targeting_solved_the_score_equation(self, result) -> None:
        assert result.validation.score_check().passed

    def test_the_contrast_is_the_difference_of_the_means(self, result) -> None:
        difference = result["ey_regime[treat if W1 > 0]"].psi - result["ey_regime[never]"].psi
        assert result["ate_regime[treat if W1 > 0 vs never]"].psi == pytest.approx(
            difference, abs=1e-12
        )

    def test_a_rule_that_ignores_W_equals_the_static_fit(self, frame) -> None:
        constant = fit(
            frame,
            interventions=(
                Static(0, name="never"),
                Rule(lambda w: np.ones(len(w), dtype=int), name="everyone"),
            ),
        )
        static = fit(frame, interventions=(Static(0, name="never"), Static(1, name="everyone")))
        assert constant["ey_regime[everyone]"].psi == static["ey_regime[everyone]"].psi

    def test_the_reference_regime_can_be_chosen(self, frame) -> None:
        flipped = fit(
            frame,
            reference="treat if W1 > 0",
            interventions=(
                Static(0, name="never"),
                Rule(lambda w: (np.asarray(w["W1"]) > 0).astype(int), name="treat if W1 > 0"),
            ),
        )
        assert "ate_regime[never vs treat if W1 > 0]" in flipped.estimates


class TestAStochasticRegime:
    def test_a_coin_flip_regime_lands_midway_between_the_arms(self, frame) -> None:
        r"""``g* = (1/2, 1/2)`` means :math:`\Psi = \tfrac12 E[Y^1] + \tfrac12 E[Y^0]`.

        Not an exact identity in finite samples: the two fits solve different score
        equations, so their targeted predictions differ by the fluctuation. The
        comparison is therefore against the arm fit's midpoint within a standard error --
        which is enough to catch a regime that fell back to one column, since the two
        arms here are far more than an SE apart.
        """
        arms = fit(frame, estimands=("ey0", "ey1"))
        coin = fit(
            frame,
            interventions=(
                Stochastic(lambda w: np.column_stack([np.full(len(w), 0.5)] * 2), "coin"),
            ),
        )
        midpoint = 0.5 * (arms["ey1"].psi + arms["ey0"].psi)
        estimate = coin["ey_regime[coin]"]
        assert estimate.psi == pytest.approx(midpoint, abs=estimate.std_error)
        assert abs(arms["ey1"].psi - arms["ey0"].psi) > 4 * estimate.std_error

    def test_it_is_refused_when_the_density_is_not_normalised(self, frame) -> None:
        with pytest.raises(DataError, match="rows summing to"):
            fit(
                frame,
                interventions=(
                    Stochastic(lambda w: np.column_stack([np.full(len(w), 0.5)] * 2) * 3, "bad"),
                ),
            )


class TestWhatIsRefused:
    def test_an_arm_estimand_cannot_be_asked_of_a_regime_fit(self, frame) -> None:
        with pytest.raises(ValueError, match="do not belong to a fit"):
            fit(frame, estimands=("ey1",), interventions=(Static(0), Static(1)))

    def test_a_regime_estimand_cannot_be_asked_without_interventions(self, frame) -> None:
        with pytest.raises(ValueError, match="do not belong to a fit"):
            fit(frame, estimands=("ey_regime",))

    def test_an_unknown_reference_regime_is_refused(self, frame) -> None:
        with pytest.raises(DataError, match="is not one of the regimes"):
            fit(frame, reference="nowhere", interventions=(Static(0), Static(1)))


class TestTheRegimesTravelWithTheFit:
    @pytest.fixture(scope="class")
    def result(self, frame):
        return fit(
            frame,
            interventions=(
                Static(0, name="never"),
                Rule(lambda w: (np.asarray(w["W1"]) > 0).astype(int), name="rule"),
            ),
        )

    def test_support_is_reported_per_regime(self, result) -> None:
        report = result.sensitivity.support()
        assert set(report.regimes) == {"never", "rule"}
        assert report.regimes["rule"].min_support_propensity > 0.0
        assert "rule" in report.summary()

    def test_support_is_refused_on_an_arm_fit(self, frame) -> None:
        with pytest.raises(ValueError, match="declared none"):
            fit(frame).sensitivity.support()

    def test_a_truncation_sweep_retargets_the_same_regimes(self, result) -> None:
        curve = result.sensitivity.truncation_curve([0.01, 0.05])
        names = {str(value) for value in curve["estimand"]}
        assert names == set(result.estimates)

    def test_a_round_trip_preserves_every_number(self, result, tmp_path) -> None:
        path = tmp_path / "regime.npz"
        result.save(path)
        loaded = load(path)
        for name, estimate in result.estimates.items():
            assert loaded.estimates[name].psi == estimate.psi
            assert np.array_equal(loaded.estimates[name].influence_curve, estimate.influence_curve)
        # The evaluated densities travel, so everything reached through retarget still
        # targets the regimes the fit declared -- even though the rule is long gone.
        assert loaded.nuisance.regimes is not None
        np.testing.assert_array_equal(
            loaded.nuisance.regimes.values, result.nuisance.regimes.values
        )
        assert loaded.sensitivity.support().regimes.keys() == {"never", "rule"}

    def test_a_rule_cannot_be_rebuilt_from_the_recipe(self, result) -> None:
        """Honest rather than convenient: a callable is not describable by a recipe."""
        from cleverly.estimators.recipe import TMLERecipe

        recipe = TMLERecipe.from_estimator(result.estimator)
        assert not recipe.learners_reconstructible
        with pytest.raises(ValueError, match="rule or a stochastic density"):
            recipe.build()

    def test_static_regimes_can_be_rebuilt(self, frame) -> None:
        from cleverly.estimators.recipe import TMLERecipe

        result = fit(frame, interventions=(Static(0), Static(1)))
        rebuilt = TMLERecipe.from_estimator(result.estimator).build()
        assert [item.level for item in rebuilt.interventions] == [0, 1]
