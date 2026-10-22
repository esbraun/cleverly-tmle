"""The pre-run design test of ``refuter-calibration-point`` (X82 plan 8.2 to 8.5).

Every defect of a power cell is sized here from the declared law (AGENT-BRIEF "Power-cell
sizing"), and the count rules are checked against the operating characteristics the plan
quotes. Nothing here fits an estimator.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import stats

from tests.studies import refuter_calibration_point as study
from tests.studies import refuter_calibration_point_properties as properties

LAW = study.quadrature()

#: The placebo shift, in placebo SD, that gives power 0.95 at ``R = 20`` (plan 8.2).
PLACEBO_SHIFT_FLOOR = 0.850
#: The confounded control's contrast, in control SE, that gives power 0.95 (plan 8.2).
CONTROL_SHIFT_FLOOR = 3.605


class TestLaw:
    def test_the_binary_prevalence_is_near_one_tenth(self) -> None:
        assert study.prevalence(LAW) == pytest.approx(0.100, abs=0.002)

    def test_the_gaussian_ate_is_the_declared_effect(self) -> None:
        assert study.truth("gaussian", LAW) == pytest.approx(study.GAUSSIAN_EFFECT, abs=1e-10)

    def test_the_quadrature_agrees_with_a_large_draw(self) -> None:
        frame = study.draw_law("binomial", 200_000, seed=1)
        assert frame["Y"].mean() == pytest.approx(study.prevalence(LAW), abs=0.004)
        assert frame["A"].mean() == pytest.approx(LAW.mean(LAW.g), abs=0.005)

    def test_the_declared_mechanism_is_the_law_with_u_integrated_out(self) -> None:
        frame = study.draw_law("binomial", 50_000, seed=2)
        # Within a narrow band of the declared propensity, the treated share matches it.
        band = frame["p1"].between(0.45, 0.55)
        assert frame.loc[band, "A"].mean() == pytest.approx(frame.loc[band, "p1"].mean(), abs=0.015)
        assert np.allclose(frame["p0"] + frame["p1"], 1.0)

    def test_the_valid_control_has_no_contrast_and_the_confounded_one_has(self) -> None:
        assert study.control_shift("binomial", 0.0, law=LAW) == pytest.approx(0.0, abs=1e-12)
        assert study.control_shift("gaussian", 0.0, law=LAW) == pytest.approx(0.0, abs=1e-12)
        assert study.control_shift("binomial", 1.0, law=LAW) > 1.0


class TestDefectSizing:
    @pytest.mark.parametrize("family", ["gaussian", "binomial"])
    def test_the_stale_share_moves_the_placebo_by_0_85_sd(self, family: str) -> None:
        assert study.placebo_shift(family, study.STALE_SHARE[family], law=LAW) >= (
            PLACEBO_SHIFT_FLOOR
        )

    @pytest.mark.parametrize("family", ["gaussian", "binomial"])
    def test_a_fresh_placebo_has_no_shift(self, family: str) -> None:
        assert study.placebo_shift(family, 0.0, law=LAW) == pytest.approx(0.0, abs=1e-12)

    def test_a_fully_stale_placebo_reads_the_ate(self) -> None:
        sd = study.placebo_shift("gaussian", 1.0, law=LAW) / study.truth("gaussian", LAW)
        assert sd > 10.0

    @pytest.mark.parametrize("family", ["gaussian", "binomial"])
    def test_the_control_delta_moves_the_control_by_3_605_se(self, family: str) -> None:
        assert study.control_shift(family, study.CONTROL_DELTA[family], law=LAW) >= (
            CONTROL_SHIFT_FLOOR
        )

    def test_the_placebo_floor_gives_power_0_95_at_the_default_refits(self) -> None:
        r = study.REFITS
        noncentrality = PLACEBO_SHIFT_FLOOR * math.sqrt(r)
        critical = stats.t.ppf(1.0 - study.LEVEL / 2.0, r - 1)
        power = stats.nct.sf(critical, r - 1, noncentrality) + stats.nct.cdf(
            -critical, r - 1, noncentrality
        )
        assert power >= 0.949

    def test_the_control_floor_gives_power_0_95(self) -> None:
        z = stats.norm.ppf(1.0 - study.LEVEL / 2.0)
        power = stats.norm.cdf(CONTROL_SHIFT_FLOOR - z) + stats.norm.cdf(-CONTROL_SHIFT_FLOOR - z)
        assert power >= 0.95

    @pytest.mark.parametrize("factor", [0.5, 2.0])
    def test_a_halved_or_doubled_se_is_detected_with_power_0_98(self, factor: float) -> None:
        df = study.REFITS - 1
        low, high = stats.chi2.ppf([study.LEVEL / 2.0, 1.0 - study.LEVEL / 2.0], df)
        # Under the null, (R - 1) rho^2 ~ chi^2_{R-1}; the mutation divides rho by ``factor``.
        scale = 1.0 / factor**2
        power = stats.chi2.cdf(low / scale, df) + stats.chi2.sf(high / scale, df)
        assert power >= 0.98


class TestCountRules:
    def test_the_interval_thresholds_reproduce_the_count_rules(self) -> None:
        assert properties.count_rule_agrees()

    @pytest.mark.parametrize(
        ("size", "expected"), [(0.05, 0.9879), (0.08, 0.6634), (0.10, 0.2849), (0.15, 0.0043)]
    )
    def test_the_size_rule_operating_characteristic(self, size: float, expected: float) -> None:
        passed = stats.binom.cdf(study.SIZE_MAX_REJECTIONS, study.SIZE_TRIALS, size)
        assert passed == pytest.approx(expected, abs=5e-4)

    @pytest.mark.parametrize(
        ("power", "expected"), [(0.95, 1.0), (0.80, 0.8139), (0.70, 0.2229), (0.50, 1.53e-4)]
    )
    def test_the_power_rule_operating_characteristic(self, power: float, expected: float) -> None:
        passed = stats.binom.sf(study.POWER_MIN_DETECTIONS - 1, study.POWER_TRIALS, power)
        assert passed == pytest.approx(expected, abs=5e-4)


class TestCells:
    def test_the_cells_are_the_plan_cells(self) -> None:
        assert set(study.CELLS) == {"T1", "T2", "T3", "C1", "D1", "D2"}
        full = {name for name, cell in study.CELLS.items() if cell.readings == study.READINGS_ALL}
        assert full == {"T1", "T2", "D1"}
        assert len(study.PROPERTY_CELLS[study.SIZE_FAMILY]) == 12
        assert len(study.PROPERTY_CELLS[study.POWER_FAMILY]) == 15

    def test_every_power_trial_is_a_size_trial(self) -> None:
        assert study.POWER_TRIALS <= study.SIZE_TRIALS


class TestMutations:
    def test_the_stale_patch_keeps_the_declared_share_and_restores(self) -> None:
        import importlib

        module = importlib.import_module("cleverly.validation.refute")
        original = module._permuted_treatment
        treatment = np.r_[np.ones(500), np.zeros(1500)]
        data = SimpleNamespace(treatment=treatment)
        with study.stale_rows(0.35):
            placebo = module._permuted_treatment(data, np.random.default_rng(0))
        assert module._permuted_treatment is original
        assert placebo.sum() == treatment.sum()
        # At least the kept share agrees; the permuted rest agrees at the marginal rate.
        agreement = float(np.mean(placebo == treatment))
        assert agreement >= 0.35 + 0.65 * 0.6

    def test_the_rescaled_subset_reading_is_the_reading_at_the_rescaled_se(self) -> None:
        from cleverly.validation._readings import subset_reading

        rng = np.random.default_rng(3)
        values = rng.normal(scale=0.02, size=study.REFITS)
        original = subset_reading(values, std_error=0.03, n=2000, m=1400)
        doubled = subset_reading(values, std_error=0.06, n=2000, m=1400)
        mutated = study.rescaled_subset(original.rho, 2.0)
        assert mutated.statistic == pytest.approx(doubled.rho, rel=1e-12)
        assert mutated.pvalue == pytest.approx(doubled.pvalue, rel=1e-10)
