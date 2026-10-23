"""The interpretation gate of the refutation tests: readings, registry and default cost.

Each reading has a witness (an exact or computed value) and a deliberate-mutation control that
fails when its component is wrong. The registry tests use a covering row that the test builds,
because Part A ships no row whose rule needs a study.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from itertools import combinations
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression

from cleverly import ATE, CausalStudy, PointTreatment
from cleverly.datasets import make_binary_outcome
from cleverly.exceptions import CapabilityError
from cleverly.validation import BootstrapMeasurementError, EmpiricalInclusionRule, refute
from cleverly.validation import _interpretation as interpretation
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
from tests.unit.test_generated_outcome_refutation import _eligible_result, _linear_ate_fit


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
        "replicates": 5,
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
            nodes=frozenset({None}),
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

    def test_the_monte_carlo_truth_variance_widens_each_coverage_interval(self) -> None:
        # A draw at 2 SE misses the exact truth and is a hit once V_T widens its interval.
        psi, se = np.full(10, 2.0) + np.linspace(0.0, 0.01, 10), np.ones(10)
        exact = band_readings(psi, se, 0.0, EquivalenceBands())[2]
        sampled = band_readings(psi, se, 0.0, EquivalenceBands(), truth_variance=0.1)[2]
        assert exact.estimate == 0.0 and sampled.estimate == 1.0


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

    def test_an_outside_fit_names_the_nearest_row_not_the_first(self) -> None:
        """The sentence reads the row that differs on the fewest axes."""
        far = replace(
            _covering_row("placebo", "t"),
            scope=CalibrationScope(
                methods=frozenset({"LTMLE"}), outcome_families=frozenset({"survival"})
            ),
        )
        near = _covering_row("placebo", "t")
        fit = _fit_configuration(method="CTMLE")
        for rows in ((far, near), (near, far)):
            sentence = resolve("placebo", "t", fit, evidence=rows).sentence
            assert "LTMLE" not in sentence and "survival" not in sentence
            assert "method 'CTMLE'" in sentence

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
                nodes=frozenset({None}),
                replicates=frozenset({5}),
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

    def test_a_combination_no_cell_ran_is_a_finding(self) -> None:
        row = self._row()
        base = _fit_configuration()
        crossed = replace(
            row,
            scope=replace(
                row.scope,
                methods=frozenset({"TMLE", "CTMLE"}),
                outcome_families=frozenset({"gaussian", "binomial"}),
            ),
        )
        cells = {
            "s1": CalibrationCell("size", True, base),
            "s2": CalibrationCell(
                "size", True, replace(base, method="CTMLE", outcome_family="binomial")
            ),
            "p1": CalibrationCell("power", True, base),
            "p2": CalibrationCell(
                "power", True, replace(base, method="CTMLE", outcome_family="binomial")
            ),
        }
        crossed = replace(crossed, study=("study", "s1", "s2", "p1", "p2"))
        findings = check_scope_containment(crossed, cells)
        assert any("TMLE" in item and "binomial" in item for item in findings)
        assert any("CTMLE" in item and "gaussian" in item for item in findings)

    def test_learners_are_checked_per_fit_not_pooled(self) -> None:
        row, base = self._row(), _fit_configuration()
        other = replace(base, learners=("Ridge()", "Lasso()"))
        pooled = replace(
            row,
            scope=replace(row.scope, learners=frozenset({*base.learners, *other.learners})),
            study=("study", "s1", "s2", "p1", "p2"),
        )
        cells = {
            "s1": CalibrationCell("size", True, base),
            "s2": CalibrationCell("size", True, other),
            "p1": CalibrationCell("power", True, base),
            "p2": CalibrationCell("power", True, other),
        }
        assert check_scope_containment(pooled, cells)

    def test_a_row_at_one_replicate_count_covers_no_other(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        row = _covering_row("placebo", "t")
        row = replace(row, scope=replace(row.scope, replicates=frozenset({20})))
        monkeypatch.setattr(interpretation, "VERDICT_EVIDENCE", (row,))
        result = _binary_fit(cross_fit=False)
        at_five = refute(result, tests=("placebo",), n_replicates=5, random_state=3)
        at_twenty = refute(result, tests=("placebo",), n_replicates=20, random_state=3)
        assert at_five["placebo"].passed is None
        assert at_twenty["placebo"].passed is not None

    def test_event_and_cluster_bounds_must_sit_inside_the_cells(self) -> None:
        row, base = self._row(), _fit_configuration(min_event_count=30, clusters=40)
        cells = {
            "s1": CalibrationCell("size", True, base),
            "p1": CalibrationCell("power", True, base),
        }
        unbounded_events = replace(row, scope=replace(row.scope, min_event_count_min=None))
        assert check_scope_containment(unbounded_events, cells)
        wide_clusters = replace(
            row, scope=replace(row.scope, min_event_count_min=30, clusters_min=10)
        )
        assert check_scope_containment(wide_clusters, cells)
        inside = replace(row, scope=replace(row.scope, min_event_count_min=30, clusters_min=40))
        assert check_scope_containment(inside, cells) == []


class TestNegativeControlConfiguration:
    def test_the_control_outcome_family_is_the_scope_axis(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A Gaussian fit with a binary control: the control's family is binomial.
        row = _covering_row("negative_control_outcome", "z")
        row = replace(row, scope=replace(row.scope, outcome_families=frozenset({"binomial"})))
        monkeypatch.setattr(interpretation, "VERDICT_EVIDENCE", (row,))
        result = _linear_ate_fit(ATE())
        rng = np.random.default_rng(0)
        binary_control = (rng.uniform(size=result.data.n) < 0.3).astype(float)
        report = refute(
            result,
            tests=("negative_control_outcome",),
            negative_control_outcome=binary_control,
            random_state=5,
        )
        assert report["negative_control_outcome"].passed is not None
        gaussian_only = replace(
            row, scope=replace(row.scope, outcome_families=frozenset({"gaussian"}))
        )
        monkeypatch.setattr(interpretation, "VERDICT_EVIDENCE", (gaussian_only,))
        withheld = refute(
            result,
            tests=("negative_control_outcome",),
            negative_control_outcome=binary_control,
            random_state=5,
        )
        assert withheld["negative_control_outcome"].passed is None


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

    def test_the_covariate_part_check_reads_the_law_and_the_contrast(self) -> None:
        from cleverly.validation._point_generated import PointTarget

        weights = np.ones(3)
        constant = {1.0: np.array([2.0, 3.0, 4.0]), 0.0: np.array([1.0, 2.0, 3.0])}
        varying = {1.0: np.array([2.0, 3.5, 4.0]), 0.0: np.array([1.0, 2.0, 3.0])}
        contrast = PointTarget("difference", "all", 1.0, 0.0)
        assert contrast.covariate_part_vanishes(constant, weights)
        assert not contrast.covariate_part_vanishes(varying, weights)
        # A level or a ratio reads the arm functions themselves, which vary here.
        assert not PointTarget("level", "all", 1.0, None).covariate_part_vanishes(constant, weights)
        assert not PointTarget("rr", "all", 1.0, 0.0).covariate_part_vanishes(constant, weights)
        flat = {1.0: np.full(3, 2.0), 0.0: np.full(3, 1.0)}
        assert PointTarget("level", "all", 1.0, None).covariate_part_vanishes(flat, weights)
        assert PointTarget("rr", "all", 1.0, 0.0).covariate_part_vanishes(flat, weights)

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
    """A check that needs more than 20 refits is opt-in (Eric, 2026-10-22 01:51Z)."""

    def test_the_default_set_has_no_check_above_twenty_refits(self) -> None:
        assert set(DEFAULT_TESTS) == {"placebo", "random_common_cause", "subset"}
        for name in DEFAULT_TESTS:
            result = _binary_fit(cross_fit=False)
            counter = _CountingEstimator(result.estimator)
            result = replace(result, estimator=counter)
            refute(result, tests=(name,), random_state=1)
            assert counter.calls <= 20, (name, counter.calls)

    def test_a_default_set_with_a_baseline_refit_per_draw_would_fail(self) -> None:
        result = _binary_fit(cross_fit=False)
        counter = _CountingEstimator(result.estimator)
        result = replace(result, estimator=counter)
        refute(result, tests=("random_common_cause",), seed_spread=True, random_state=1)
        assert counter.calls > 20  # the opt-in path costs more, so the default must not use it

    def test_a_default_set_with_the_dummy_outcome_would_fail(self) -> None:
        result = _linear_ate_fit(ATE())
        counter = _CountingEstimator(result.estimator)
        result = replace(result, estimator=counter)
        refute(result, tests=("dummy_outcome",), n_replicates=30, random_state=1)
        assert counter.calls > 20  # so the dummy outcome must stay out of the default set

    def test_the_generated_outcome_tests_are_not_default(self) -> None:
        assert "dummy_outcome" not in DEFAULT_TESTS
        assert "simulated_outcome" not in DEFAULT_TESTS


class TestBootstrapFoldRule:
    def test_every_copy_of_a_unit_keeps_the_original_units_fold(self) -> None:
        from cleverly.validation.refute import _bootstrap_design, _CopyFolds

        result = _binary_fit(cross_fit=True, n=200)
        data = result.data
        design = _bootstrap_design(data, n_replicates=3, resampling="iid", random_state=4)
        draws = {draw.replicate: draw for draw in design.draws}
        folds = _CopyFolds(result, data, design, draws)
        original = np.asarray(result.repeats[0].folds.assignment)
        for replicate in draws:
            sample = folds.sample(replicate)
            index, _ = design.indices(data, draws[replicate])
            assert sample.n == index.size
            frozen = folds.options(replicate)["frozen_folds"]
            assert len(frozen) == len(result.repeats)
            _, expected = np.unique(original[index], return_inverse=True)
            np.testing.assert_array_equal(frozen[0].assignment, expected)
            # The mutation: a refit that redrew folds would split copies of one unit apart.
            copies = np.flatnonzero(np.bincount(index, minlength=200) > 1)
            assert copies.size > 0
            for unit in copies:
                positions = np.flatnonzero(index == unit)
                assert len(set(frozen[0].assignment[positions])) == 1

    @staticmethod
    def _run(monkeypatch: pytest.MonkeyPatch | None = None) -> tuple[Any, _CountingEstimator]:
        result = _binary_fit(cross_fit=True, n=200)
        counter = _CountingEstimator(result.estimator)
        result = replace(result, estimator=counter)
        refute(
            result,
            tests=("bootstrap_measurement_error",),
            bootstrap_measurement_error=BootstrapMeasurementError(("W1",)),
            measurement_error_rule=EmpiricalInclusionRule(alpha=0.5, minimum_draws=4),
            n_replicates=4,
            random_state=5,
        )
        return result, counter

    def test_the_estimator_receives_the_original_folds_of_the_copied_rows(self) -> None:
        result, counter = self._run()
        assert len(counter.kwargs) == 4
        for options in counter.kwargs:
            frozen = options["frozen_folds"]
            assert len(frozen) == len(result.repeats)
            assert frozen[0].n == result.data.n

    def test_dropping_the_fold_options_leaves_no_frozen_folds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cleverly.validation.refute import _CopyFolds

        monkeypatch.setattr(_CopyFolds, "options", lambda self, replicate: {})
        _, counter = self._run()
        assert all("frozen_folds" not in options for options in counter.kwargs)


class TestBootstrapStability:
    def _run(self, **options: Any) -> tuple[Any, _CountingEstimator, Any]:
        result = _binary_fit(cross_fit=True, n=200)
        counter = _CountingEstimator(result.estimator)
        result = replace(result, estimator=counter)
        report = refute(
            result, tests=("bootstrap_stability",), n_replicates=4, random_state=5, **options
        )
        return result, counter, report

    def test_it_is_opt_in_and_descriptive(self) -> None:
        assert "bootstrap_stability" not in DEFAULT_TESTS
        _, _, report = self._run()
        test = report["bootstrap_stability"]
        assert test.passed is None
        assert test.rung == "L0"
        assert len(test.values) == 4

    def test_every_refit_reuses_the_original_seeds_and_folds(self) -> None:
        result, counter, _ = self._run()
        seeds = [item.seed for item in result.repeats]
        for options in counter.kwargs:
            assert list(options["frozen_seeds"]) == seeds
            assert len(options["frozen_folds"]) == len(result.repeats)

    def test_the_shift_is_the_mean_against_the_original_fit(self) -> None:
        _, _, report = self._run()
        test = report["bootstrap_stability"]
        shift = dict(test.extras)["shift against the original fit"]
        assert shift == pytest.approx(np.mean(test.values) - test.original)

    def test_a_ratio_reads_on_the_log_scale(self) -> None:
        from cleverly import RiskRatio

        frame, _ = make_binary_outcome(n=200, seed=3)
        study = CausalStudy(
            frame,
            design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3")),
        )
        result = study.identify(RiskRatio()).estimate(
            outcome_learner=LogisticRegression(max_iter=1000),
            treatment_learner=LogisticRegression(max_iter=1000),
            n_folds=2,
        )
        estimand = next(name for name in result.estimates if name.startswith("rr"))
        test = refute(
            result,
            estimand=estimand,
            tests=("bootstrap_stability",),
            n_replicates=4,
            random_state=5,
        )["bootstrap_stability"]
        shift = dict(test.extras)["shift against the original fit"]
        expected = np.mean(np.log(test.values)) - np.log(test.original)
        assert shift == pytest.approx(expected)
        assert test.mean_kind == "geometric" and test.spread_kind == "log"

    def test_one_draw_is_refused(self) -> None:
        result = _binary_fit(cross_fit=True, n=200)
        with pytest.raises(CapabilityError, match="at least 2"):
            refute(result, tests=("bootstrap_stability",), n_replicates=1, random_state=5)


class TestLearnedRulePlacebo:
    """The placebo law makes every rule's value ``E[Y]`` (X82 plan 5.1.1)."""

    @staticmethod
    def _fit() -> Any:
        from tests.unit import _learned_rule_support as support

        return support.fit()

    def test_the_placebo_reads_the_value_against_the_outcome_mean(self) -> None:
        from tests.unit import _learned_rule_support as support

        result = self._fit()
        report = refute(
            result, estimand=support.NAME, tests=("placebo",), n_replicates=4, random_state=1
        )
        test = report["placebo"]
        mean = float(np.mean(result.data.outcome))
        assert test.expectation == f"~ {mean:g}"
        assert "outcome mean" in test.detail

    def test_the_other_refit_tests_meet_no_learned_rule_stop(self) -> None:
        """X82 part F4 holds the realized rules on every other refit test."""
        from cleverly.validation.refute import refute_refusal
        from tests.unit import _learned_rule_support as support

        tests = ("placebo", "random_common_cause", "subset", "bootstrap_stability")
        assert refute_refusal(self._fit(), estimand=support.NAME, tests=tests) is None

    def test_the_reference_is_exact_over_every_permutation(self) -> None:
        """An IPW value of a fixed rule, with the marginal ``g``, averages to the weighted mean."""
        import itertools
        from types import SimpleNamespace

        from cleverly.validation.refute import _learned_rule_reference

        y = np.array([1.0, 0.0, 2.0, 3.0, 1.5, 0.5])
        w = np.array([1.0, 2.0, 1.0, 3.0, 1.0, 2.0])
        a = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
        rule = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
        share = {1.0: a.mean(), 0.0: 1.0 - a.mean()}
        values = []
        for perm in set(itertools.permutations(a)):
            permuted = np.array(perm)
            g = np.array([share[rule[i]] for i in range(6)])
            values.append(np.sum(w * (permuted == rule) * y / g) / np.sum(w))
        data = SimpleNamespace(outcome=y, weights=w, observed=np.ones(6, dtype=bool))
        assert np.mean(values) == pytest.approx(_learned_rule_reference(data))
        # The mutation: an unweighted mean of a weighted fit is not the placebo value.
        assert np.mean(values) != pytest.approx(float(np.mean(y)))


def _msm_stub(design: np.ndarray, weights: np.ndarray, q: np.ndarray, mass: np.ndarray) -> Any:
    """A finite law as an MSM fit: rows are support points, ``mass`` their probability."""
    from types import SimpleNamespace

    from cleverly.msm import MSMSet

    arms = (0.0, 1.0)
    terms = tuple(f"t{index}" for index in range(design.shape[2]))
    model = MSMSet(terms, design, weights, arms)  # type: ignore[arg-type]
    nuisance = SimpleNamespace(
        msm=model,
        propensity=SimpleNamespace(arm=lambda arm: np.full(len(q), 0.5)),
        scaler=SimpleNamespace(unscale_level=lambda values: values),
    )
    targeted = SimpleNamespace(arms={0.0: q, 1.0: q})
    return SimpleNamespace(
        config=SimpleNamespace(parameter_axis="msm"),
        nuisance=nuisance,
        fluctuations={"msm": SimpleNamespace(targeted=targeted)},
        data=SimpleNamespace(weights=mass),
    )


class TestMSMPlacebo:
    """Under the placebo law ``m(a, W) = q(W)`` for every arm (X82 plan 5.1.4)."""

    V = np.array([-1.0, 0.0, 0.5, 2.0])
    MASS = np.array([0.1, 0.4, 0.3, 0.2])
    Q = np.array([0.2, 0.9, -0.4, 1.3])

    def _design(self, columns: Callable[[float, np.ndarray], list[np.ndarray]]) -> np.ndarray:
        return np.stack([np.column_stack(columns(a, self.V)) for a in (0.0, 1.0)], axis=1)

    def test_a_tensor_design_with_a_product_weight_is_exactly_zero(self) -> None:
        from cleverly.validation.refute import _msm_reference

        design = self._design(lambda a, v: [np.ones_like(v), np.full_like(v, a), v, a * v])
        weights = np.outer(1.0 + (self.V > 0), [0.6, 1.8])
        stub = _msm_stub(design, weights, self.Q, self.MASS)
        for term in ("t1", "t3"):
            reference = _msm_reference(stub, f"msm[{term}]")
            assert reference is not None and reference.exact and reference.value == 0.0
        # The covariate terms carry the projection of q, so they are not claimed exact.
        assert not _msm_reference(stub, "msm[t2]").exact

    def test_a_design_without_the_arm_intercept_has_a_computed_nonzero_null(self) -> None:
        from cleverly.validation.refute import _msm_reference

        design = self._design(lambda a, v: [np.ones_like(v), np.full_like(v, a), a * v])
        weights = np.ones((4, 2))
        stub = _msm_stub(design, weights, self.Q, self.MASS)
        reference = _msm_reference(stub, "msm[t2]")
        assert reference is not None and not reference.exact
        # Independent weighted least squares over the enumerated (unit, arm) cells.
        rows = design.reshape(8, 3)
        target = np.repeat(self.Q, 2)
        root = np.sqrt(np.repeat(self.MASS, 2))
        beta = np.linalg.lstsq(rows * root[:, None], target * root, rcond=None)[0]
        assert reference.value == pytest.approx(beta[2], abs=1e-12)
        # The mutation: reading this coefficient against 0 would be wrong by beta[2].
        assert abs(beta[2]) > 0.1

    def test_a_non_product_weight_loses_the_exact_zero(self) -> None:
        from cleverly.validation.refute import _msm_reference

        design = self._design(lambda a, v: [np.ones_like(v), np.full_like(v, a), v])
        weights = np.column_stack([np.ones(4), 1.0 + self.V**2])
        stub = _msm_stub(design, weights, self.Q, self.MASS)
        assert not _msm_reference(stub, "msm[t1]").exact

    def test_a_real_fit_reads_the_treatment_coefficient_at_zero(self) -> None:
        from tests.unit._simulated_confounding_support import _fit_msm

        result = _fit_msm()
        exact = refute(
            result, estimand="msm[treatment]", tests=("placebo",), n_replicates=4, random_state=1
        )["placebo"]
        assert exact.expectation == "~ 0"
        assert exact.statistic is not None
        computed = refute(
            result, estimand="msm[baseline]", tests=("placebo",), n_replicates=4, random_state=1
        )["placebo"]
        assert computed.passed is None and computed.statistic is None
        assert "computed projection" in computed.detail

    def test_a_computed_reference_states_no_verdict_under_a_covering_row(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The gate, not only the reference: a covering row must not license a computed null.

        The mutation is a version that builds the ``t`` reading against the computed
        projection. That version prints a verdict here and fails. The exact-zero fit under
        the same row is the positive control: the row does license the ``t`` rule.
        """
        from cleverly import MSMProjection
        from cleverly.msm import MSM
        from tests.unit._simulated_confounding_support import _estimate, _fit_msm, _study

        row = _covering_row("placebo", "t")
        row = replace(
            row,
            scope=replace(
                row.scope, methods=frozenset({"TMLE"}), estimand_kinds=frozenset({"msm[treatment]"})
            ),
        )
        monkeypatch.setattr(interpretation, "VERDICT_EVIDENCE", (row,))
        model = MSM(
            design=lambda a, w: np.column_stack(
                [
                    np.ones(len(w)),
                    np.full(len(w), float(a == 1)),
                    float(a == 1) * np.asarray(w["W"]),
                ]
            ),
            terms=("intercept", "treatment", "treated_w"),
            design_kind="known",
        )
        computed = refute(
            _estimate(_study(), MSMProjection(model)),
            estimand="msm[treatment]",
            tests=("placebo",),
            n_replicates=4,
            random_state=1,
        )["placebo"]
        assert computed.passed is None and computed.statistic is None
        exact = refute(
            _fit_msm(),
            estimand="msm[treatment]",
            tests=("placebo",),
            n_replicates=4,
            random_state=1,
        )["placebo"]
        assert exact.passed is not None and exact.rung == "L2"

    def test_the_negative_control_still_refuses_an_msm_coefficient(self) -> None:
        from tests.unit._simulated_confounding_support import _fit_msm

        result = _fit_msm()
        with pytest.raises(CapabilityError, match="no universal null"):
            refute(
                result,
                estimand="msm[treatment]",
                tests=("negative_control_outcome",),
                negative_control_outcome=np.zeros(result.data.n),
            )


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
        assert len(report["placebo"].values) == 20
        with pytest.raises(ValueError, match="placebo_mode"):
            refute(result, tests=("placebo",), placebo_mode="shuffle")


class _IgnoreLast(BaseEstimator, RegressorMixin):
    """Fit logistic regression on the first three columns only."""

    def fit(self, x: Any, y: Any) -> _IgnoreLast:
        self.model_ = LogisticRegression(max_iter=1000).fit(x[:, :3], y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, x: Any) -> Any:
        return self.model_.predict_proba(x[:, :3])

    def predict(self, x: Any) -> Any:
        return self.model_.predict(x[:, :3])


class TestPairedNoise:
    def test_a_learner_that_ignores_the_noise_column_gives_exactly_zero(self) -> None:
        frame, _ = make_binary_outcome(n=300, seed=5)
        study = CausalStudy(
            frame,
            design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3")),
        )
        result = study.identify(ATE()).estimate(
            outcome_learner=_IgnoreLast(),
            treatment_learner=_IgnoreLast(),
            n_folds=2,
            simultaneous=False,
        )
        report = refute(result, tests=("random_common_cause",), n_replicates=3, random_state=1)
        assert report["random_common_cause"].statistic == 0.0

    @pytest.mark.parametrize("random_state", [None, 7])
    def test_repeated_cross_fitting_keeps_every_repeat_seed(self, random_state: int | None) -> None:
        from tests.conftest import fast_tmle

        frame, _ = make_binary_outcome(n=300, seed=5)
        result = (
            fast_tmle(
                repeats=2,
                n_folds=2,
                cross_fit=True,
                outcome_learner=_IgnoreLast(),
                treatment_learner=_IgnoreLast(),
                random_state=random_state,
            )
            .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"])
            .single()
        )
        report = refute(result, tests=("random_common_cause",), n_replicates=2, random_state=1)
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


class TestFrozenSplitsAndCovariatePart:
    def test_frozen_folds_meet_the_trainability_preflight(self) -> None:
        from cleverly.exceptions import DataError
        from cleverly.learners.crossfit import Folds

        result = _binary_fit(cross_fit=True)
        treatment = np.asarray(result.data.treatment)
        labels = (treatment == 1).astype(np.int64)
        with pytest.raises(DataError, match="training complement"):
            result.estimator.refit(
                result.data, frozen_folds=[Folds(labels, 2)] * len(result.repeats)
            )

    def test_a_varying_effect_would_withhold_the_ratio_and_coverage(self) -> None:
        from cleverly.validation._point_generated import PointTarget

        # Only the rows of the population count: a contrast that varies on a row outside
        # the ATT population still has a zero covariate part there.
        target = PointTarget("difference", "value", 1.0, 0.0)
        means = {1.0: np.array([2.0, 2.0, 9.0]), 0.0: np.array([1.0, 1.0, 1.0])}
        assert target.covariate_part_vanishes(means, np.array([1.0, 1.0, 0.0]))
        assert not target.covariate_part_vanishes(means, np.ones(3))

    def test_the_registered_processes_carry_a_scalar_effect(self) -> None:
        # The covariate-part check reads the law, so a tau(W) process would be read
        # correctly; the catalog still declares a scalar effect only.
        from cleverly.validation import GaussianAdjustmentOutcome

        with pytest.raises((ValueError, TypeError)):
            GaussianAdjustmentOutcome(effect=lambda w: w[:, 0])  # type: ignore[arg-type]
