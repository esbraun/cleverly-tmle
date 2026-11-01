"""Point generated processes beyond the Gaussian ATE (X82 part E1).

The Bernoulli and bounded processes, per-arm effects, multi-arm treatments, arm levels and
ratios. Each truth is checked against a closed form computed here, and each sign, mask and
guard has a mutation control.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import (
    ATC,
    ATE,
    ATT,
    CausalStudy,
    CounterfactualMean,
    OddsRatio,
    PointTreatment,
    RiskRatio,
)
from cleverly.exceptions import CapabilityError, RefusalReason
from cleverly.validation import (
    BernoulliAdjustmentOutcome,
    BoundedAdjustmentOutcome,
    GaussianAdjustmentOutcome,
    refute,
)
from cleverly.validation._point_generated import PlasmodeLaw, PointTarget, arm_means
from cleverly.validation.outcome_processes import arm_term

REFUTE = sys.modules["cleverly.validation.refute"]


def _expit(values: Any) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-np.asarray(values, dtype=float))))


def _frame(n: int, *, arms: int = 2, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    covariates = rng.normal(size=(n, 2))
    if arms == 2:
        treatment = (rng.random(n) < _expit(0.5 * covariates[:, 0])).astype(int)
    else:
        treatment = rng.integers(0, arms, size=n)
    return pd.DataFrame(
        {
            "W1": covariates[:, 0],
            "W2": covariates[:, 1],
            "A": treatment,
            "Yb": (rng.random(n) < 0.4).astype(float),
            "Yc": rng.random(n),
            "Yg": rng.normal(size=n),
        }
    )


def _fit(frame: pd.DataFrame, outcome: str, estimand: Any, **options: Any) -> Any:
    study = CausalStudy(
        frame, design=PointTreatment(outcome=outcome, treatment="A", adjustment=("W1", "W2"))
    )
    learner = LogisticRegression() if outcome == "Yb" else LinearRegression()
    options.setdefault("cross_fit", False)
    return study.identify(estimand).estimate(
        outcome_learner=learner,
        treatment_learner=LogisticRegression(max_iter=1000),
        random_state=1,
        **options,
    )


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return _frame(300)


@pytest.fixture(scope="module")
def three_arm() -> pd.DataFrame:
    return _frame(300, arms=3, seed=5)


#: An identity-link risk whose arm contrast does not vary with W.
IDENTITY = BernoulliAdjustmentOutcome(
    intercept=0.4, effect=0.15, adjustment_scale=0.3, link="identity"
)


class TestCatalog:
    def test_arm_effects_map_labels_to_terms(self) -> None:
        process = GaussianAdjustmentOutcome(arm_effects={"a": 0.0, "b": 0.5, "c": -1.0})
        terms = arm_term(
            process, np.array([2.0, 0.0, 1.0]), arm_codes=(0.0, 1.0, 2.0), levels=("a", "b", "c")
        )
        np.testing.assert_array_equal(terms, [-1.0, 0.0, 0.5])

    def test_an_arm_without_an_effect_is_refused(self) -> None:
        process = GaussianAdjustmentOutcome(arm_effects={0: 0.0, 1: 0.5})
        with pytest.raises(ValueError, match="no effect for the declared arm"):
            arm_term(process, np.zeros(2), arm_codes=(0.0, 1.0, 2.0), levels=(0, 1, 2))

    def test_the_bounded_draws_stay_inside_and_have_the_declared_mean(self) -> None:
        process = BoundedAdjustmentOutcome(precision=5.0)
        mean = np.full(200_000, 3.4)
        draws = process.sample(np.random.default_rng(0), mean, bounds=(2.0, 6.0))
        assert draws.min() >= 2.0 and draws.max() <= 6.0
        # Beta(m phi, (1 - m) phi) has variance m (1 - m) / (phi + 1) on the unit scale.
        unit_sd = np.sqrt(0.35 * 0.65 / 6.0)
        assert abs(draws.mean() - 3.4) < 4 * 4.0 * unit_sd / np.sqrt(mean.size)

    def test_independence_reads_the_arm_terms(self) -> None:
        assert BernoulliAdjustmentOutcome(effect=0.0, adjustment_scale=0.0).independent
        assert BernoulliAdjustmentOutcome(
            adjustment_scale=0.0, arm_effects={0: 0.3, 1: 0.3}
        ).independent
        assert not BernoulliAdjustmentOutcome(
            adjustment_scale=0.0, arm_effects={0: 0.0, 1: 0.3}
        ).independent


class TestTruths:
    """Each truth against a closed form, with the mutation that breaks it."""

    covariates = np.array([[-1.0, 0.5], [0.0, -0.5], [1.0, 0.0], [2.0, 1.0]])

    def _data(self, treatment: Any, weights: Any = None) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(
            arm_codes=(0.0, 1.0),
            treatment_levels=(0, 1),
            treatment=np.asarray(treatment, dtype=float),
            weights=np.ones(4) if weights is None else np.asarray(weights, dtype=float),
            covariates=self.covariates,
        )

    def _logit_means(self, process: BernoulliAdjustmentOutcome) -> dict[float, np.ndarray]:
        # Worked by hand: the standardized covariate sum over root count, then expit.
        z = (self.covariates - self.covariates.mean(0)) / self.covariates.std(0)
        shift = process.adjustment_scale * z.sum(1) / np.sqrt(2.0)
        return {
            code: _expit(process.intercept + shift + process.effect * code) for code in (0.0, 1.0)
        }

    def test_the_logit_ate_is_the_mean_risk_difference(self) -> None:
        process = BernoulliAdjustmentOutcome(intercept=-0.3, effect=0.8, adjustment_scale=1.2)
        data = self._data([0, 1, 0, 1])
        target = PointTarget("difference", "all", 1.0, 0.0)
        means = arm_means(process, self.covariates, target.codes, data)
        expected = self._logit_means(process)
        truth = target.truth(means, np.ones(4))
        assert truth == pytest.approx(float(np.mean(expected[1.0] - expected[0.0])), abs=1e-14)
        # Mutation: the reversed key reads the negated truth, so a swapped direction fails.
        swapped = PointTarget("difference", "all", 0.0, 1.0).truth(means, np.ones(4))
        assert swapped == pytest.approx(-truth) and swapped != pytest.approx(truth)

    def test_att_and_atc_sum_over_their_own_arm(self) -> None:
        process = BernoulliAdjustmentOutcome(intercept=-0.3, effect=0.8, adjustment_scale=1.2)
        data = self._data([0, 1, 0, 1])
        expected = self._logit_means(process)
        difference = expected[1.0] - expected[0.0]
        for kind, rows in (("value", [1, 3]), ("reference", [0, 2])):
            target = PointTarget("difference", kind, 1.0, 0.0)  # type: ignore[arg-type]
            means = arm_means(process, self.covariates, target.codes, data)
            weights = REFUTE.held_weights(target, data)
            assert target.truth(means, weights) == pytest.approx(float(np.mean(difference[rows])))
            # Mutation: every row is the ATE, not this population's contrast.
            assert target.truth(means, np.ones(4)) != pytest.approx(
                float(np.mean(difference[rows]))
            )

    def test_the_fit_weights_enter_the_level(self) -> None:
        process = BernoulliAdjustmentOutcome(intercept=-0.3, effect=0.8, adjustment_scale=1.2)
        weights = np.array([1.0, 3.0, 0.5, 2.0])
        data = self._data([0, 1, 0, 1], weights)
        target = PointTarget("level", "all", 1.0, None)
        means = arm_means(process, self.covariates, target.codes, data)
        expected = self._logit_means(process)[1.0]
        truth = target.truth(means, REFUTE.held_weights(target, data))
        assert truth == pytest.approx(float(np.average(expected, weights=weights)))
        # Mutation: dropping the weights gives the unweighted level, a different number.
        assert truth != pytest.approx(float(np.mean(expected)))

    def test_ratios_are_ratios_of_levels(self) -> None:
        process = BernoulliAdjustmentOutcome(intercept=-0.3, effect=0.8, adjustment_scale=1.2)
        data = self._data([0, 1, 0, 1])
        expected = self._logit_means(process)
        one, zero = float(np.mean(expected[1.0])), float(np.mean(expected[0.0]))
        means = arm_means(process, self.covariates, (1.0, 0.0), data)
        rr = PointTarget("rr", "all", 1.0, 0.0).truth(means, np.ones(4))
        odds = PointTarget("or", "all", 1.0, 0.0).truth(means, np.ones(4))
        assert rr == pytest.approx(one / zero)
        assert odds == pytest.approx((one / (1 - one)) / (zero / (1 - zero)))

    def test_per_arm_gaussian_contrasts_are_the_declared_differences(self) -> None:
        from types import SimpleNamespace

        process = GaussianAdjustmentOutcome(arm_effects={0: 0.0, 1: 0.5, 2: -1.0})
        data = SimpleNamespace(arm_codes=(0.0, 1.0, 2.0), treatment_levels=(0, 1, 2))
        target = PointTarget("difference", "all", 2.0, 1.0)
        means = arm_means(process, self.covariates, target.codes, data)
        assert target.truth(means, np.ones(4)) == pytest.approx(-1.5)
        assert target.covariate_part_vanishes(means, np.ones(4))

    def test_the_identity_link_contrast_is_constant_and_the_logit_one_is_not(self) -> None:
        data = self._data([0, 1, 0, 1])
        target = PointTarget("difference", "all", 1.0, 0.0)
        identity = arm_means(IDENTITY, self.covariates, target.codes, data)
        assert target.truth(identity, np.ones(4)) == pytest.approx(0.15)
        assert target.covariate_part_vanishes(identity, np.ones(4))
        logit = arm_means(BernoulliAdjustmentOutcome(), self.covariates, target.codes, data)
        assert not target.covariate_part_vanishes(logit, np.ones(4))


class TestPlasmodeLaw:
    def test_a_three_arm_model_is_floored_and_normalized(self, three_arm: pd.DataFrame) -> None:
        result = _fit(three_arm, "Yg", ATE())
        law = PlasmodeLaw.fit(result)
        assert law.probabilities.shape == (300, 3)
        np.testing.assert_allclose(law.probabilities.sum(axis=1), 1.0)
        lower = law.bounds[0] if law.bounds is not None else 0.0
        assert law.floor >= lower / (1.0 + 3 * lower) - 1e-12

    def test_the_att_truth_is_the_exact_support_sum(self, frame: pd.DataFrame) -> None:
        result = _fit(frame, "Yb", ATT())
        law = PlasmodeLaw.fit(result)
        process = BernoulliAdjustmentOutcome(intercept=-0.3, effect=0.8, adjustment_scale=1.2)
        target = PointTarget("difference", "value", 1.0, 0.0)
        means, weights, _ = law.means(process, result.data, target)
        exact = target.truth(means, weights)
        # Monte Carlo of the law itself: units drawn from the support, arms from the frozen g.
        rng = np.random.default_rng(11)
        rows = rng.integers(0, result.data.n, size=400_000)
        treated = rng.random(rows.size) < law.treated[rows]
        contrast = means[1.0][rows] - means[0.0][rows]
        estimate = float(np.mean(contrast[treated]))
        spread = float(np.std(contrast[treated]) / np.sqrt(np.sum(treated)))
        assert abs(estimate - exact) < 4 * spread
        # Mutation: weighting by the observed treated rows is the held design, not this law.
        observed = np.asarray(result.data.treatment == 1.0, dtype=float)
        assert abs(target.truth(means, observed) - exact) > 4 * spread


class TestEndToEnd:
    """Real fits with GLM learners at R = 400, the calibrated minimum."""

    def test_a_binary_identity_link_reads_three_bands_on_the_held_design(
        self, frame: pd.DataFrame
    ) -> None:
        report = refute(
            _fit(frame, "Yb", ATE()),
            tests=("simulated_outcome",),
            simulated_outcome=IDENTITY,
            n_replicates=400,
            random_state=4,
        )
        test = report["simulated_outcome"]
        assert [reading.name for reading in test.readings] == ["bias", "se_ratio", "coverage"]
        assert test.rung == "L2" and test.passed is True
        assert test.declared_effect == pytest.approx(0.15)

    def test_a_binary_logit_link_reads_bias_only(self, frame: pd.DataFrame) -> None:
        test = refute(
            _fit(frame, "Yb", ATE()),
            tests=("simulated_outcome",),
            simulated_outcome=BernoulliAdjustmentOutcome(),
            n_replicates=400,
            random_state=4,
        )["simulated_outcome"]
        assert [reading.name for reading in test.readings] == ["bias"]
        assert "withheld" in test.detail and test.passed is True

    def test_a_risk_ratio_reads_on_the_log_scale(self, frame: pd.DataFrame) -> None:
        report = refute(
            _fit(frame, "Yb", RiskRatio()),
            estimand="rr",
            tests=("simulated_outcome", "plasmode"),
            simulated_outcome=BernoulliAdjustmentOutcome(),
            n_replicates=400,
            random_state=4,
        )
        assert report["simulated_outcome"].rung == "L2"
        plasmode = report["plasmode"]
        assert "on the log scale" in plasmode.detail
        assert plasmode.passed is True and len(plasmode.readings) == 3

    def test_reading_a_ratio_on_the_natural_scale_fails(
        self, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Mutation: compare the ratio itself with its log-scale standard error.
        monkeypatch.setattr(PointTarget, "ratio", property(lambda self: False))
        plasmode = refute(
            _fit(frame, "Yb", RiskRatio()),
            estimand="rr",
            tests=("plasmode",),
            simulated_outcome=BernoulliAdjustmentOutcome(),
            n_replicates=400,
            random_state=4,
        )["plasmode"]
        assert plasmode.passed is False
        assert not plasmode.readings[1].inside

    def test_an_arm_level_and_an_odds_ratio_run(self, frame: pd.DataFrame) -> None:
        level = refute(
            _fit(frame, "Yb", CounterfactualMean(treatment=1)),
            estimand="ey1",
            tests=("plasmode",),
            simulated_outcome=BernoulliAdjustmentOutcome(),
            n_replicates=400,
            random_state=4,
        )["plasmode"]
        assert level.rung == "L2" and level.passed is True
        odds = refute(
            _fit(frame, "Yb", OddsRatio()),
            estimand="or",
            tests=("simulated_outcome",),
            simulated_outcome=BernoulliAdjustmentOutcome(),
            n_replicates=400,
            random_state=4,
        )["simulated_outcome"]
        assert odds.rung == "L2" and odds.passed is True

    def test_a_three_arm_att_reads_its_pair(self, three_arm: pd.DataFrame) -> None:
        report = refute(
            _fit(three_arm, "Yg", ATT()),
            estimand="att[2.0 vs 0.0]",
            tests=("simulated_outcome", "plasmode"),
            simulated_outcome=GaussianAdjustmentOutcome(arm_effects={0: 0.0, 1: 0.5, 2: -1.0}),
            n_replicates=400,
            random_state=6,
        )
        for name in ("simulated_outcome", "plasmode"):
            assert report[name].declared_effect == pytest.approx(-1.0)
            assert report[name].passed is True and len(report[name].readings) == 3
        assert "smallest arm probability" in report["plasmode"].scope_sentence

    def test_a_bounded_outcome_runs_on_a_q_bounds_fit(self, frame: pd.DataFrame) -> None:
        process = BoundedAdjustmentOutcome(
            intercept=0.45, effect=0.1, adjustment_scale=0.3, link="identity"
        )
        test = refute(
            _fit(frame, "Yc", ATC(), q_bounds=(0.0, 1.0)),
            estimand="atc",
            tests=("simulated_outcome",),
            simulated_outcome=process,
            n_replicates=400,
            random_state=4,
        )["simulated_outcome"]
        assert test.declared_effect == pytest.approx(0.1)
        assert test.passed is True and len(test.readings) == 3
        assert "phi 20" in test.scope_sentence

    def test_a_nonpositive_refitted_ratio_is_a_retained_failure(
        self, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = _fit(frame, "Yb", RiskRatio())
        original = REFUTE._run_empirical_refits

        def with_a_negative(*args: Any, **kwargs: Any) -> Any:
            from dataclasses import replace

            records, failures = original(*args, **kwargs)
            return (replace(records[0], estimate=-0.5), *records[1:]), failures

        monkeypatch.setattr(REFUTE, "_run_empirical_refits", with_a_negative)
        test = refute(
            result,
            estimand="rr",
            tests=("simulated_outcome",),
            simulated_outcome=BernoulliAdjustmentOutcome(),
            n_replicates=20,
            random_state=4,
        )["simulated_outcome"]
        assert len(test.failures) == 1 and "not positive" in test.failures[0].message
        assert test.passed is None


class TestRefusals:
    """Each request E1 keeps refused, before any refit."""

    def test_a_gaussian_process_on_a_q_bounds_fit(self, frame: pd.DataFrame) -> None:
        with pytest.raises(CapabilityError, match="BoundedAdjustmentOutcome") as caught:
            refute(
                _fit(frame, "Yc", ATE(), q_bounds=(0.0, 1.0)),
                tests=("simulated_outcome",),
                n_replicates=10,
            )
        assert caught.value.refusal.reason is RefusalReason.INVALID_DECLARATION

    def test_a_bounded_process_without_q_bounds(self, frame: pd.DataFrame) -> None:
        with pytest.raises(CapabilityError, match="declares none"):
            refute(
                _fit(frame, "Yg", ATE()),
                tests=("simulated_outcome",),
                simulated_outcome=BoundedAdjustmentOutcome(),
                n_replicates=10,
            )

    def test_a_process_of_another_family(self, frame: pd.DataFrame) -> None:
        with pytest.raises(CapabilityError, match="family") as caught:
            refute(
                _fit(frame, "Yg", ATE()),
                tests=("simulated_outcome",),
                simulated_outcome=BernoulliAdjustmentOutcome(),
                n_replicates=10,
            )
        assert caught.value.refusal.reason is RefusalReason.IMPLEMENTATION_GAP

    def test_an_identity_link_that_leaves_the_unit_interval(self, frame: pd.DataFrame) -> None:
        with pytest.raises(CapabilityError, match=r"leaves \[0, 1\]"):
            refute(
                _fit(frame, "Yb", ATE()),
                tests=("plasmode",),
                simulated_outcome=BernoulliAdjustmentOutcome(
                    intercept=0.9, effect=0.3, link="identity"
                ),
                n_replicates=10,
            )

    def test_a_dummy_process_that_reads_the_treatment(self, frame: pd.DataFrame) -> None:
        with pytest.raises(CapabilityError, match="effect=0 and adjustment_scale=0"):
            refute(
                _fit(frame, "Yb", ATE()),
                tests=("dummy_outcome",),
                dummy_outcome=BernoulliAdjustmentOutcome(),
                n_replicates=10,
            )

    def test_an_independent_bernoulli_dummy_runs(self, frame: pd.DataFrame) -> None:
        test = refute(
            _fit(frame, "Yb", ATE()),
            tests=("dummy_outcome",),
            dummy_outcome=BernoulliAdjustmentOutcome(effect=0.0, adjustment_scale=0.0),
            n_replicates=10,
            random_state=1,
        )["dummy_outcome"]
        assert test.declared_effect == 0.0 and test.rung == "L1"

    def test_arm_effects_on_a_longitudinal_panel_are_refused(self) -> None:
        from tests.unit.test_longitudinal_generated_outcomes import CONTRAST, _panel
        from tests.unit.test_longitudinal_generated_outcomes import _fit as _panel_fit

        with pytest.raises(CapabilityError, match="arm_effects"):
            refute(
                _panel_fit(_panel()),
                estimand=CONTRAST,
                tests=("simulated_outcome",),
                simulated_outcome=GaussianAdjustmentOutcome(arm_effects={0: 0.0, 1: 1.0}),
                n_replicates=10,
            )

    def test_a_bernoulli_process_on_a_longitudinal_panel_is_refused(self) -> None:
        from tests.unit.test_longitudinal_generated_outcomes import CONTRAST, _panel
        from tests.unit.test_longitudinal_generated_outcomes import _fit as _panel_fit

        with pytest.raises(CapabilityError, match="Gaussian processes only"):
            refute(
                _panel_fit(_panel()),
                estimand=CONTRAST,
                tests=("simulated_outcome",),
                simulated_outcome=BernoulliAdjustmentOutcome(),
                n_replicates=10,
            )


class TestRegistry:
    def test_levels_and_ratios_resolve_to_the_rows_of_the_ate(self, frame: pd.DataFrame) -> None:
        """Nit of the E plan review: no new row, the exact rows cover every point target."""
        from cleverly.validation import _interpretation

        rules = ("bias_band", "se_ratio_band", "coverage_band")
        resolved: dict[str, dict[tuple[str, str], Any]] = {}
        for estimand, alias in (
            (ATE(), "ate"),
            (RiskRatio(), "rr"),
            (CounterfactualMean(1), "ey1"),
        ):
            fit = _interpretation.fit_configuration(_fit(frame, "Yb", estimand), alias)
            resolved[alias] = {
                (instrument, rule): _interpretation.resolve(instrument, rule, fit).evidence
                for instrument in ("generated_outcome", "plasmode")
                for rule in rules
            }
        assert all(row is not None for row in resolved["ate"].values())
        assert resolved["rr"] == resolved["ate"] == resolved["ey1"]
