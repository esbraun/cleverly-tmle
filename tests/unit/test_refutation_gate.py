"""The interpretation gate of the refutation tests: readings, registry and default cost.

Each reading has a witness (an exact or computed value) and a deliberate-mutation control that
fails when its component is wrong. The registry tests use a covering row that the test builds,
because Part A ships no row whose rule needs a study.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from cleverly import ATE, CausalStudy, PointTreatment
from cleverly.datasets import make_binary_outcome
from cleverly.validation import _interpretation as interpretation
from cleverly.validation import refute
from cleverly.validation._interpretation import (
    CalibrationCell,
    CalibrationScope,
    FitConfiguration,
    VerdictEvidence,
    check_scope_containment,
    learner_fingerprint,
    resolve,
)
from cleverly.validation._readings import (
    EquivalenceBands,
    band_readings,
    paired_shift,
    placebo_reading,
    subset_reading,
    subset_reference_sd,
)
from cleverly.validation.refute import DEFAULT_TESTS
from tests.unit.test_generated_outcome_refutation import _eligible_result


def _fit_configuration(**overrides: Any) -> FitConfiguration:
    base = {
        "method": "TMLE",
        "estimand_kind": "ate",
        "outcome_family": "gaussian",
        "treatment_kind": "discrete",
        "learners": ("LinearRegression()", "LogisticRegression()"),
        "cross_fitting": "off",
        "inference": "influence_curve",
        "known_mechanism": False,
        "n": 400,
        "covariates_per_row": 0.01,
        "min_event_count": None,
        "clusters": None,
        "arms": 2,
    }
    base.update(overrides)
    return FitConfiguration(**base)


def _covering_row(instrument: str, rule: str) -> VerdictEvidence:
    """A row that covers a stub fit, for a test that needs the verdict rule to run."""
    return VerdictEvidence(
        instrument=instrument,
        rule=rule,
        rung="L2",
        theory="test row",
        scope=CalibrationScope(
            methods=frozenset({"TMLE"}),
            estimand_kinds=frozenset({"ate"}),
            outcome_families=frozenset({"gaussian", "binomial"}),
            treatment_kinds=frozenset({"discrete"}),
            learners=None,
            cross_fitting=frozenset({"off", "folds=2", "folds=3"}),
            inference=frozenset({"influence_curve"}),
            known_mechanism=frozenset({False, True}),
            arms=frozenset({2}),
            n_min=1,
        ),
        sentence="covered by the test row",
        study=("test-study", "size", "power"),
    )


def _binary_fit(*, cross_fit: bool = True, n: int = 400, seed: int = 3, **options: Any) -> Any:
    frame, _ = make_binary_outcome(n=n, seed=seed)
    study = CausalStudy(
        frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3")),
    )
    return study.identify(ATE()).estimate(
        outcome_learner=LogisticRegression(max_iter=1000),
        treatment_learner=LogisticRegression(max_iter=1000),
        cross_fit=cross_fit,
        n_folds=2,
        simultaneous=False,
        random_state=seed,
        **options,
    )


class TestSubsetReference:
    """The reference SD of a subset refit is exact for the Horvitz-Thompson statistic."""

    @staticmethod
    def _enumerated(n: int, m: int) -> tuple[float, float]:
        rng = np.random.default_rng(5)
        z = rng.normal(size=n) + 2.0  # A Y / g at a known g, one value per unit
        se = float(z.std(ddof=0) / np.sqrt(n))
        means = np.array([z[list(idx)].mean() for idx in combinations(range(n), m)])
        return se, float(means.std(ddof=0))

    @pytest.mark.parametrize(("n", "m"), [(10, 6), (11, 7), (12, 5)])
    def test_the_reference_equals_the_exact_enumerated_sd(self, n: int, m: int) -> None:
        se, exact = self._enumerated(n, m)
        assert subset_reference_sd(se, n, m) == pytest.approx(exact, rel=1e-12)

    def test_the_old_reference_fails_the_witness(self) -> None:
        se, exact = self._enumerated(12, 8)
        # The rule before X82 compared with SE * sqrt(1/f), which is 1.22 times too large at
        # f = 2/3 and misses the exact SD by far more than the float tolerance.
        assert se * np.sqrt(12 / 8) != pytest.approx(exact, rel=1e-3)

    def test_a_reference_with_f_in_place_of_m_over_n_fails_when_they_differ(self) -> None:
        n, m, fraction = 12, 7, 0.6  # int(0.6 * 12) = 7, so m / n = 0.5833 != 0.6
        se, exact = self._enumerated(n, m)
        wrong = se * np.sqrt((1 / fraction - 1) * n / (n - 1))
        assert wrong != pytest.approx(exact, rel=1e-6)

    def test_the_scatter_ratio_is_one_at_the_reference(self) -> None:
        n, m = 200, 140
        reference = subset_reference_sd(0.1, n, m)
        values = np.linspace(-1.0, 1.0, 20)
        values = values / values.std(ddof=1) * reference
        reading = subset_reading(values, std_error=0.1, n=n, m=m)
        assert reading.rho == pytest.approx(1.0, rel=1e-12)
        assert reading.pvalue > 0.5

    def test_both_tails_read(self) -> None:
        n, m = 200, 140
        reference = subset_reference_sd(0.1, n, m)
        base = np.linspace(-1.0, 1.0, 20)
        base = base / base.std(ddof=1)
        assert subset_reading(base * 3 * reference, std_error=0.1, n=n, m=m).pvalue < 1e-4
        assert subset_reading(base * reference / 3, std_error=0.1, n=n, m=m).pvalue < 1e-4


class TestPlaceboReading:
    def test_a_null_centred_sample_has_a_large_pvalue(self) -> None:
        rng = np.random.default_rng(0)
        reading = placebo_reading(rng.normal(size=20), std_error=1.0)
        assert reading.kind == "t"
        assert reading.df == 19
        assert not reading.rejects

    def test_size_is_near_nominal(self) -> None:
        rng = np.random.default_rng(1)
        rejections = np.mean(
            [placebo_reading(rng.normal(size=20), std_error=1.0).rejects for _ in range(2000)]
        )
        assert abs(rejections - 0.05) < 0.02

    def test_a_shifted_sample_rejects(self) -> None:
        rng = np.random.default_rng(2)
        assert placebo_reading(rng.normal(size=20) + 1.5, std_error=1.0).rejects

    def test_a_constant_zero_deviation_is_descriptive(self) -> None:
        reading = placebo_reading(np.zeros(20), std_error=0.1)
        assert reading.kind == "constant_zero"
        assert not reading.rejects
        assert np.isnan(reading.t)

    def test_a_constant_nonzero_deviation_rejects_without_dividing_by_zero(self) -> None:
        reading = placebo_reading(np.full(20, 0.3), std_error=0.1)
        assert reading.kind == "constant_nonzero"
        assert reading.rejects
        assert np.isposinf(reading.t)

    def test_the_floor_scales_with_the_standard_error(self) -> None:
        tiny = np.array([0.0, 1e-9, -1e-9, 0.0, 1e-9, -1e-9])
        assert placebo_reading(tiny, std_error=1e-6).kind == "t"
        assert placebo_reading(tiny, std_error=1e3).kind == "constant_zero"


class TestPairedShift:
    def test_equal_refits_give_exactly_zero(self) -> None:
        values = np.array([1.0, 2.0, 3.0])
        shift, spread, _ = paired_shift(values, values, std_error=0.5)
        assert shift == 0.0
        assert spread == 0.0

    def test_a_shift_is_in_standard_errors(self) -> None:
        shift, _, _ = paired_shift([1.2, 1.4], [1.0, 1.0], std_error=0.1)
        assert shift == pytest.approx(3.0)


class TestEquivalenceBands:
    @staticmethod
    def _joint_pass(draws: int, *, bias: float = 0.0, ratio: float = 1.0, trials: int) -> float:
        rng = np.random.default_rng(20261022)
        bands = EquivalenceBands()
        passed = 0
        for _ in range(trials):
            psi = bias + rng.standard_normal(draws)
            se = ratio * np.sqrt(rng.chisquare(50, draws) / 50)
            readings = band_readings(psi, se, 0.0, bands)
            passed += all(reading.inside for reading in readings)
        return passed / trials

    def test_a_calibrated_estimator_passes_at_400_and_not_at_300(self) -> None:
        assert self._joint_pass(400, trials=600) >= 0.88
        assert self._joint_pass(300, trials=600) < 0.88

    def test_a_stub_at_se_ratio_097_passes(self) -> None:
        assert self._joint_pass(400, ratio=0.97, trials=600) >= 0.80

    @pytest.mark.parametrize(("bias", "ratio"), [(0.0, 0.85), (0.0, 1.2), (0.35, 1.0)])
    def test_calibrated_defects_fail(self, bias: float, ratio: float) -> None:
        assert self._joint_pass(400, bias=bias, ratio=ratio, trials=300) < 0.05

    def test_the_old_exact_null_rule_fails_a_correct_estimator(self) -> None:
        rng = np.random.default_rng(3)
        failures = 0
        for _ in range(200):
            psi = rng.standard_normal(400)
            se = np.sqrt(rng.chisquare(50, 400) / 50) * 0.97
            ratio = band_readings(psi, se, 0.0, EquivalenceBands())[1]
            failures += not (ratio.lower <= 1.0 <= ratio.upper)
        assert failures > 20  # "the interval contains 1" rejects a 3% miscalibration

    def test_the_monte_carlo_truth_variance_widens_the_bias_interval(self) -> None:
        rng = np.random.default_rng(4)
        psi, se = rng.standard_normal(400), np.ones(400)
        narrow = band_readings(psi, se, 0.0, EquivalenceBands())[0]
        wide = band_readings(psi, se, 0.0, EquivalenceBands(), truth_variance=0.5)[0]
        assert wide.upper - wide.lower > narrow.upper - narrow.lower


class TestRegistry:
    def test_every_non_exact_row_names_a_study(self) -> None:
        for row in interpretation.VERDICT_EVIDENCE:
            assert row.exact or row.study, row

    def test_a_row_without_study_or_exact_reference_cannot_claim_l2(self) -> None:
        with pytest.raises(ValueError, match="without a registered study"):
            VerdictEvidence(
                "placebo", "t", "L2", "x", CalibrationScope(methods=frozenset({"TMLE"})), "s"
            )

    def test_only_an_exact_row_may_be_unrestricted(self) -> None:
        with pytest.raises(ValueError, match="unrestricted scope"):
            VerdictEvidence(
                "placebo",
                "t",
                "L1",
                "x",
                CalibrationScope.unrestricted(),
                "s",
            )

    def test_a_rule_with_no_row_is_descriptive(self) -> None:
        resolution = resolve("placebo", "t", _fit_configuration())
        assert not resolution.licensed
        assert resolution.rung == "L0"

    def test_a_covering_row_licenses_and_an_outside_fit_does_not(self) -> None:
        rows = (_covering_row("placebo", "t"),)
        assert resolve("placebo", "t", _fit_configuration(), evidence=rows).licensed
        other = resolve("placebo", "t", _fit_configuration(method="CTMLE"), evidence=rows)
        assert not other.licensed
        assert "method" in other.sentence
        assert "TMLE" in other.sentence

    @pytest.mark.parametrize(
        "change",
        [
            {"outcome_family": "survival"},
            {"treatment_kind": "continuous"},
            {"cross_fitting": "folds=10"},
            {"inference": "few_cluster_plugin"},
            {"known_mechanism": None},
            {"arms": 3},
        ],
    )
    def test_each_categorical_axis_gates(self, change: dict[str, Any]) -> None:
        row = _covering_row("placebo", "t")
        if "known_mechanism" in change:
            row = replace(row, scope=replace(row.scope, known_mechanism=frozenset({False})))
            change = {"known_mechanism": True}
        assert row.scope.covers(_fit_configuration())
        assert not row.scope.covers(_fit_configuration(**change))

    def test_the_data_shape_axes_gate(self) -> None:
        scope = replace(
            _covering_row("placebo", "t").scope,
            n_min=500,
            n_max=10_000,
            covariates_per_row_max=0.05,
            min_event_count_min=20,
            clusters_min=None,
        )
        fit = _fit_configuration(n=800, min_event_count=30)
        assert scope.covers(fit)
        assert not scope.covers(replace(fit, n=400))
        assert not scope.covers(replace(fit, n=20_000))
        assert not scope.covers(replace(fit, covariates_per_row=0.2))
        assert not scope.covers(replace(fit, min_event_count=5))
        assert not scope.covers(replace(fit, min_event_count=None))
        assert not scope.covers(replace(fit, clusters=12))

    def test_a_booster_resolves_early_stopping_with_the_row_count(self) -> None:
        from sklearn.ensemble import HistGradientBoostingRegressor

        learner = HistGradientBoostingRegressor()
        assert "early_stopping=False" in learner_fingerprint(learner, 9_000)
        assert "early_stopping=True" in learner_fingerprint(learner, 12_000)
        assert learner_fingerprint(None, 100) == "default-library"

    def test_a_real_fit_reads_its_configuration(self) -> None:
        result = _binary_fit(cross_fit=True)
        fit = interpretation.fit_configuration(result, "ate")
        assert fit.method == "TMLE"
        assert fit.outcome_family == "binomial"
        assert fit.cross_fitting == "folds=2"
        assert fit.inference == "influence_curve"
        assert fit.arms == 2
        assert fit.min_event_count is not None and fit.min_event_count > 0
        assert not fit.known_mechanism


class TestScopeContainment:
    @staticmethod
    def _cells() -> dict[str, CalibrationCell]:
        fit = _fit_configuration()
        return {
            "s1": CalibrationCell("size", True, fit),
            "p1": CalibrationCell("power", True, fit),
        }

    @staticmethod
    def _row() -> VerdictEvidence:
        fit = _fit_configuration()
        return VerdictEvidence(
            "placebo",
            "t",
            "L2",
            "x",
            CalibrationScope(
                methods=frozenset({fit.method}),
                estimand_kinds=frozenset({fit.estimand_kind}),
                outcome_families=frozenset({fit.outcome_family}),
                treatment_kinds=frozenset({fit.treatment_kind}),
                learners=frozenset(fit.learners),
                cross_fitting=frozenset({fit.cross_fitting}),
                inference=frozenset({fit.inference}),
                known_mechanism=frozenset({False}),
                arms=frozenset({2}),
                n_min=fit.n,
                covariates_per_row_max=fit.covariates_per_row,
            ),
            "s",
            study=("study", "s1", "p1"),
        )

    def test_a_row_inside_its_cells_has_no_finding(self) -> None:
        assert check_scope_containment(self._row(), self._cells()) == []

    def test_an_exact_row_has_nothing_to_contain(self) -> None:
        assert check_scope_containment(interpretation.VERDICT_EVIDENCE[0], {}) == []

    def test_the_mutations_are_rejected(self) -> None:
        row, cells = self._row(), self._cells()
        extra_learner = replace(
            row, scope=replace(row.scope, learners=row.scope.learners | {"Booster()"})
        )
        smaller_n = replace(row, scope=replace(row.scope, n_min=row.scope.n_min - 1))
        more_covariates = replace(row, scope=replace(row.scope, covariates_per_row_max=0.5))
        for mutated in (extra_learner, smaller_n, more_covariates):
            assert check_scope_containment(mutated, cells)
        without_power = {key: cell for key, cell in cells.items() if key != "p1"}
        assert check_scope_containment(row, without_power)
        failed_size = {**cells, "s1": CalibrationCell("size", False, cells["s1"].configuration)}
        assert check_scope_containment(row, failed_size)


class TestEmptyRegistry:
    """With the registry empty, no rule states a verdict."""

    def test_every_passed_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(interpretation, "VERDICT_EVIDENCE", ())
        report = refute(
            _eligible_result(),
            tests=("dummy_outcome",),
            n_replicates=400,
            random_state=1,
        )
        assert report["dummy_outcome"].passed is None
        assert report.passed is None
        with pytest.raises(TypeError, match="descriptive"):
            bool(report)

    def test_a_real_fit_prints_no_verdict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(interpretation, "VERDICT_EVIDENCE", ())
        report = refute(_binary_fit(), random_state=2)
        assert [test.passed for test in report.tests] == [None, None, None]
        assert report.passed is None
        assert "VERDICT: none" in report.summary()


class TestVerdictThroughACoveringRow:
    def test_the_placebo_rule_fails_a_leaky_estimator_and_passes_a_sound_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = (_covering_row("placebo", "t"),)
        monkeypatch.setattr(interpretation, "VERDICT_EVIDENCE", rows)
        result = _binary_fit(cross_fit=False)
        sound = refute(result, tests=("placebo",), n_replicates=20, random_state=3)
        assert sound["placebo"].passed is True

        real_refit = result.estimator.refit

        def leaky(data: Any, **options: Any) -> Any:
            fitted = real_refit(data, **options)
            fitted.estimates["ate"] = SimpleNamespace(
                psi=fitted["ate"].psi + 0.5,
                plugin_std_error=fitted["ate"].plugin_std_error,
                inference="influence_curve",
            )
            return fitted

        monkeypatch.setattr(result.estimator, "refit", leaky)
        broken = refute(result, tests=("placebo",), n_replicates=20, random_state=3)
        assert broken["placebo"].passed is False

    def test_the_subset_rule_reads_both_tails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(interpretation, "VERDICT_EVIDENCE", (_covering_row("subset", "rho"),))
        result = _binary_fit(cross_fit=False)
        report = refute(result, tests=("subset",), n_replicates=20, random_state=4)
        assert report["subset"].passed is not None
        assert report["subset"].statistic_pvalue is not None

    def test_the_negative_control_rule_uses_z(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            interpretation,
            "VERDICT_EVIDENCE",
            (_covering_row("negative_control_outcome", "z"),),
        )
        result = _binary_fit(cross_fit=False)
        rng = np.random.default_rng(0)
        shifted = (rng.uniform(size=result.data.n) < 0.3).astype(float)
        report = refute(
            result,
            tests=("negative_control_outcome",),
            negative_control_outcome=shifted,
            random_state=5,
        )
        test = report["negative_control_outcome"]
        assert test.statistic is not None and np.isfinite(test.statistic)
        assert test.passed is (test.statistic_pvalue >= 0.05)


class TestGeneratedOutcomeBands:
    def test_the_default_budget_is_400_and_below_it_the_readings_are_l1(self) -> None:
        from cleverly.validation import DEFAULT_OUTCOME_REPLICATES

        assert DEFAULT_OUTCOME_REPLICATES == 400
        low = refute(_eligible_result(), tests=("dummy_outcome",), n_replicates=40, random_state=6)[
            "dummy_outcome"
        ]
        assert low.passed is None
        assert low.rung == "L1"
        assert "calibrated minimum" in low.scope_sentence

    def test_a_budget_below_ten_is_refused_before_any_refit(self) -> None:
        from cleverly.exceptions import CapabilityError

        with pytest.raises(CapabilityError, match="at least 10"):
            refute(_eligible_result(), tests=("dummy_outcome",), n_replicates=5)

    def test_a_sound_estimator_passes_at_400(self) -> None:
        from tests.unit.test_generated_outcome_refutation import _linear_ate_fit

        test = refute(_linear_ate_fit(ATE()), tests=("simulated_outcome",), random_state=7)[
            "simulated_outcome"
        ]
        assert test.rung == "L2"
        assert test.passed is True
        assert [reading.name for reading in test.readings] == ["bias", "se_ratio", "coverage"]
        assert all(reading.inside for reading in test.readings)

    def test_a_biased_estimator_fails_the_bias_band(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tests.unit.test_generated_outcome_refutation import _MeanDifferenceEstimator

        class Biased(_MeanDifferenceEstimator):
            def refit(self, data: Any, **options: Any) -> Any:
                fitted = super().refit(data, **options)
                fitted.estimate.psi += 0.5
                return fitted

        test = refute(
            _eligible_result(estimator=Biased()),
            tests=("dummy_outcome",),
            random_state=8,
        )["dummy_outcome"]
        assert test.passed is False
        assert not test.readings[0].inside

    def test_a_standard_error_that_is_too_small_fails_the_ratio_band(self) -> None:
        from tests.unit.test_generated_outcome_refutation import _MeanDifferenceEstimator

        class Overconfident(_MeanDifferenceEstimator):
            def refit(self, data: Any, **options: Any) -> Any:
                fitted = super().refit(data, **options)
                fitted.estimate.std_error *= 0.6
                fitted.estimate.plugin_std_error *= 0.6
                return fitted

        test = refute(
            _eligible_result(estimator=Overconfident()),
            tests=("dummy_outcome",),
            random_state=9,
        )["dummy_outcome"]
        assert test.passed is False
        assert not test.readings[1].inside

    def test_the_covariate_part_check_reads_the_effect_and_the_contrast(self) -> None:
        from cleverly.validation.refute import _covariate_part_vanishes

        assert _covariate_part_vanishes(
            difference_contrast=True, effect_varies_with_covariates=False
        )
        assert not _covariate_part_vanishes(
            difference_contrast=True, effect_varies_with_covariates=True
        )
        assert not _covariate_part_vanishes(
            difference_contrast=False, effect_varies_with_covariates=False
        )

    def test_a_non_registered_band_declaration_is_refused(self) -> None:
        from cleverly.exceptions import CapabilityError

        class Loose(EquivalenceBands):
            pass

        with pytest.raises(CapabilityError, match="EquivalenceBands"):
            refute(
                _eligible_result(),
                tests=("dummy_outcome",),
                outcome_bands=Loose(),
            )


class _CountingEstimator:
    """Wrap an estimator and count the refits that a test asks for."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls = 0
        self.kwargs: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def refit(self, data: Any, **options: Any) -> Any:
        self.calls += 1
        self.kwargs.append(options)
        return self.inner.refit(data, **options)


class TestDefaultCost:
    """A check that needs more than five refits is opt-in (Eric, 2026-10-22)."""

    def test_the_default_set_has_no_check_above_five_refits(self) -> None:
        assert set(DEFAULT_TESTS) == {"placebo", "random_common_cause", "subset"}
        for name in DEFAULT_TESTS:
            result = _binary_fit(cross_fit=False)
            counter = _CountingEstimator(result.estimator)
            result = replace(result, estimator=counter)
            refute(result, tests=(name,), random_state=1)
            assert counter.calls <= 5, (name, counter.calls)

    def test_a_default_set_with_a_baseline_refit_per_draw_would_fail(self) -> None:
        result = _binary_fit(cross_fit=False)
        counter = _CountingEstimator(result.estimator)
        result = replace(result, estimator=counter)
        refute(result, tests=("random_common_cause",), seed_spread=True, random_state=1)
        assert counter.calls > 5  # the opt-in path costs more, so the default must not use it

    def test_the_generated_outcome_tests_are_not_default(self) -> None:
        assert "dummy_outcome" not in DEFAULT_TESTS
        assert "simulated_outcome" not in DEFAULT_TESTS


class TestSubsetFoldSlice:
    def test_the_refit_keeps_each_drawn_units_original_fold(self) -> None:
        result = _binary_fit(cross_fit=True, n=300)
        counter = _CountingEstimator(result.estimator)
        result = replace(result, estimator=counter)
        report = refute(result, tests=("subset",), n_replicates=3, random_state=2)
        assert len(counter.kwargs) == 3
        for options in counter.kwargs:
            frozen = options["frozen_folds"]
            assert len(frozen) == len(result.repeats)
            assert frozen[0].n == int(0.7 * 300)
        assert report["subset"].values

    def test_the_slice_is_the_original_labels_of_the_drawn_rows(self) -> None:
        from cleverly.validation.refute import _sliced_folds

        result = _binary_fit(cross_fit=True, n=200)
        index = np.sort(np.random.default_rng(0).choice(200, size=140, replace=False))
        sliced = _sliced_folds(result, index)[0]
        original = np.asarray(result.repeats[0].folds.assignment)[index]
        np.testing.assert_array_equal(sliced.assignment, original)

    def test_a_split_plan_no_longer_blocks_the_subset_test(self) -> None:
        result = _binary_fit(cross_fit=True, n=200)
        plan = result.split_plan
        frame, _ = make_binary_outcome(n=200, seed=3)
        study = CausalStudy(
            frame,
            design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3")),
        )
        planned = study.identify(ATE()).estimate(
            outcome_learner=LogisticRegression(max_iter=1000),
            treatment_learner=LogisticRegression(max_iter=1000),
            n_folds=2,
            split_plan=plan,
            simultaneous=False,
            random_state=3,
        )
        report = refute(planned, tests=("subset",), n_replicates=3, random_state=1)
        assert len(report["subset"].values) == 3


class TestPlaceboDraw:
    def test_the_draw_mode_keeps_the_observed_support(self) -> None:
        from cleverly.validation.refute import _drawn_treatment

        data = _binary_fit(cross_fit=False).data
        rng = np.random.default_rng(0)
        for _ in range(20):
            drawn = _drawn_treatment(data, rng)
            assert set(np.unique(drawn)) == {0.0, 1.0}

    def test_a_gaussian_draw_would_leave_the_support(self) -> None:
        rng = np.random.default_rng(0)
        assert not set(np.unique(rng.standard_normal(50))) <= {0.0, 1.0}

    def test_the_mode_runs_end_to_end_and_an_unknown_mode_raises(self) -> None:
        result = _binary_fit(cross_fit=False)
        report = refute(result, tests=("placebo",), placebo_mode="draw", random_state=1)
        assert len(report["placebo"].values) == 5
        with pytest.raises(ValueError, match="placebo_mode"):
            refute(result, tests=("placebo",), placebo_mode="shuffle")


class TestPairedNoise:
    def test_a_learner_that_ignores_the_noise_column_gives_exactly_zero(self) -> None:
        from sklearn.base import BaseEstimator, RegressorMixin

        class IgnoreLast(BaseEstimator, RegressorMixin):
            """Fit logistic regression on the first four columns only."""

            def fit(self, x: Any, y: Any) -> IgnoreLast:
                self.model_ = LogisticRegression(max_iter=1000).fit(x[:, :3], y)
                self.classes_ = self.model_.classes_
                return self

            def predict_proba(self, x: Any) -> Any:
                return self.model_.predict_proba(x[:, :3])

            def predict(self, x: Any) -> Any:
                return self.model_.predict(x[:, :3])

        frame, _ = make_binary_outcome(n=300, seed=5)
        study = CausalStudy(
            frame,
            design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3")),
        )
        result = study.identify(ATE()).estimate(
            outcome_learner=IgnoreLast(),
            treatment_learner=IgnoreLast(),
            n_folds=2,
            simultaneous=False,
        )
        report = refute(result, tests=("random_common_cause",), n_replicates=3, random_state=1)
        assert report["random_common_cause"].statistic == 0.0

    def test_the_seed_spread_is_opt_in(self) -> None:
        result = _binary_fit(cross_fit=True, n=200)
        default = refute(result, tests=("random_common_cause",), n_replicates=2, random_state=1)
        labels = [label for label, _ in default["random_common_cause"].extras]
        assert not any("seeds" in label for label in labels)
        wide = refute(
            result,
            tests=("random_common_cause",),
            n_replicates=2,
            seed_spread=True,
            random_state=1,
        )
        assert any("seeds" in label for label, _ in wide["random_common_cause"].extras)
