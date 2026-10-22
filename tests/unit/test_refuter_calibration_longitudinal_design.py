"""The pre-run design test of ``refuter-calibration-longitudinal`` (X82 plan 8.2 to 8.5).

Every defect of a power cell is sized from the declared law (AGENT-BRIEF "Power-cell sizing").
The placebo shifts and the control contrasts come from the sizing probe
``tests/diagnostics/x82_sl_sizing/`` (``reading.csv``), which fits each cell's own estimator on
large samples of the defective law. The control contrasts are also checked against their exact
values. The count rules are checked against the operating characteristics the plan quotes.
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from cleverly.exceptions import CapabilityError
from cleverly.validation import refute
from cleverly.validation._markov_placebo import MarkovKernel
from tests.diagnostics.x82_sl_sizing import run as sizing
from tests.studies import refuter_calibration_longitudinal as study
from tests.studies import refuter_calibration_longitudinal_properties as properties

#: The placebo shift, in placebo SD, that gives power 0.95 at ``R = 20`` (plan 8.2).
PLACEBO_SHIFT_FLOOR = 0.850
#: The confounded control's contrast, in control SE, that gives power 0.95 (plan 8.2).
CONTROL_SHIFT_FLOOR = 3.605
#: The Monte Carlo SD of one probe draw's ratio, ``sqrt(n / SIZE)``.
DRAW_SD = math.sqrt(study.N / sizing.SIZE)
#: Each probe mean must clear its floor by this many Monte Carlo SDs of the mean.
MARGIN_SDS = 3.0

READING = pd.read_csv(sizing.HERE / "reading.csv", float_precision="round_trip")


def _probe_mean(cell: str, quantity: str) -> tuple[float, float]:
    """The probe's mean ratio of one quantity, and the Monte Carlo SD of that mean.

    The SD of one draw is the larger of its theoretical value and the spread of the draws.
    """
    rows = READING[(READING["cell"] == cell) & (READING["quantity"] == quantity)]
    assert len(rows) == sizing.DRAWS
    assert set(rows["size"]) == {sizing.SIZE}
    assert set(rows["n"]) == {study.N}
    spread = float(rows["shift"].std(ddof=1))
    return float(rows["shift"].mean()), max(DRAW_SD, spread) / math.sqrt(len(rows))


class TestLaw:
    def test_the_end_of_study_contrast_is_one_point_two(self) -> None:
        assert math.isclose(study.END_OF_STUDY_ATE, 1.2, abs_tol=1e-12)

    def test_a_large_draw_reads_the_declared_law(self) -> None:
        frame = study.draw_end_of_study(200_000, seed=1)
        for w2 in (0, 1):
            share = frame.loc[frame["W2"] == w2, "A1"].mean()
            assert share == pytest.approx(study.a1_propensity(w2), abs=0.006)
        # The valid control reads the outcome's baseline part and nothing else.
        residual = frame["NC"] - 0.5 * frame["W1"] - 0.4 * frame["W2"]
        assert float(np.corrcoef(residual, frame["A1"])[0, 1]) == pytest.approx(0.0, abs=0.01)
        assert float(np.corrcoef(residual, frame["A2"])[0, 1]) == pytest.approx(0.0, abs=0.01)

    def test_the_a1_model_is_saturated_by_a_main_terms_logit(self) -> None:
        # With U integrated out, A1 reads W only through the binary W2, so two parameters
        # (intercept, W2) fit its two values exactly. W1 carries no A1 signal.
        frame = study.draw_end_of_study(200_000, seed=2)
        gap = frame.loc[frame["W1"] > 0, "A1"].mean() - frame.loc[frame["W1"] <= 0, "A1"].mean()
        assert gap == pytest.approx(0.0, abs=0.01)

    def test_the_survival_truth_is_the_registered_law_truth(self) -> None:
        _, truth = study.make_longitudinal_survival(10, seed=0)
        expected = truth["ate_regimen[always vs never @ t=2]"]
        assert study.truth("survival") == pytest.approx(expected, abs=1e-12)

    def test_a_draw_outside_the_declared_bounds_is_refused(self) -> None:
        with pytest.raises(RuntimeError, match="q_bounds"):
            study.draw_end_of_study(1_000, seed=3, delta=50.0)


class TestDefectSizing:
    @pytest.mark.parametrize("cell", list(study.CELLS))
    def test_the_stale_share_moves_the_placebo_by_0_85_sd(self, cell: str) -> None:
        mean, sd = _probe_mean(cell, "stale")
        # The contrasts of law E are positive and the survival risk contrast is negative.
        assert abs(mean) >= PLACEBO_SHIFT_FLOOR + MARGIN_SDS * sd

    @pytest.mark.parametrize("cell", [c for c, s in study.CELLS.items() if s.law == "end_of_study"])
    def test_the_control_delta_moves_the_control_by_3_605_se(self, cell: str) -> None:
        mean, sd = _probe_mean(cell, "confounded")
        assert mean >= CONTROL_SHIFT_FLOOR + MARGIN_SDS * sd

    @pytest.mark.parametrize("cell", [c for c, s in study.CELLS.items() if s.law == "end_of_study"])
    def test_the_valid_control_reads_no_contrast(self, cell: str) -> None:
        mean, sd = _probe_mean(cell, "control")
        assert abs(mean) <= 4.0 * sd

    @pytest.mark.parametrize("cell", [c for c, s in study.CELLS.items() if s.law == "end_of_study"])
    def test_the_probe_reads_the_exact_confounded_contrast(self, cell: str) -> None:
        rows = READING[(READING["cell"] == cell) & (READING["quantity"] == "confounded")]
        error = float(rows["se_at_n"].mean()) * DRAW_SD / math.sqrt(len(rows))
        exact = study.confounded_contrast(cell)
        assert float(rows["estimate"].mean()) == pytest.approx(exact, abs=4.0 * error)
        assert exact > 0.0

    def test_the_confounded_contrast_vanishes_without_u(self) -> None:
        for cell in ("S1", "I1", "SN1"):
            assert study.confounded_contrast(cell, delta=0.0) == 0.0

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
        scale = 1.0 / factor**2
        power = stats.chi2.cdf(low / scale, df) + stats.chi2.sf(high / scale, df)
        assert power >= 0.98

    def test_the_probe_ran_at_the_declared_defect_sizes(self) -> None:
        assert set(READING["cell"]) == set(study.CELLS)
        for cell, spec in study.CELLS.items():
            rows = READING[READING["cell"] == cell]
            assert set(rows["stale_share"]) == {study.STALE_SHARE[spec.law]}
            if spec.law == "end_of_study":
                assert set(rows["control_delta"]) == {study.CONTROL_DELTA}


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
        assert set(study.CELLS) == {"L1", "S1", "I1", "B1", "B2", "B3", "SN1"}
        full = {name for name, cell in study.CELLS.items() if cell.readings == study.READINGS_ALL}
        assert full == {"S1", "I1", "B1", "B2", "B3", "SN1"}
        assert len(study.PROPERTY_CELLS[study.SIZE_FAMILY]) == 20
        assert len(study.PROPERTY_CELLS[study.POWER_FAMILY]) == 27

    def test_every_power_trial_is_a_size_trial(self) -> None:
        assert study.POWER_TRIALS <= study.SIZE_TRIALS

    def test_the_cell_configurations_are_a_committed_artifact(self) -> None:
        assert study.STUDY.property_artifacts == (properties.CELLS_FILE,)

    def test_a_survival_fit_refuses_the_negative_control(self) -> None:
        # L1 reads no control because the shipped refute refuses one on a survival panel.
        frame = study.draw_survival(study.N, seed=4)
        result = study.fit_cell("L1", frame, seed=5)
        with warnings.catch_warnings(), pytest.raises(CapabilityError, match="survival"):
            warnings.simplefilter("ignore")
            refute(
                result,
                estimand=study.CELLS["L1"].estimand,
                tests=("negative_control_outcome",),
                negative_control_outcome=np.zeros(study.N),
            )


class TestConfiguration:
    @pytest.mark.parametrize(
        ("cell", "method", "folds", "learners"),
        [
            ("L1", "LTMLE", "folds=5", 4),
            ("S1", "SDR", "folds=5", 3),
            ("I1", "IPSI", "folds=5", 3),
            ("B1", "SequentialGComputation", "off", 2),
            ("B2", "SequentialIPW", "off", 1),
            ("B3", "ParametricGFormula", "off", 3),
            ("SN1", "StructuralNestedMean", "folds=5", 3),
        ],
    )
    def test_a_trial_records_the_fitted_configuration(
        self, cell: str, method: str, folds: str, learners: int
    ) -> None:
        frame = study.draw_law(study.CELLS[cell].law, study.N, seed=6)
        result = study.fit_cell(cell, frame, seed=7)
        configuration = study.trial_configuration(result, study.CELLS[cell].estimand)
        assert configuration["method"] == method
        assert configuration["cross_fitting"] == folds
        assert configuration["replicates"] == study.REFITS
        assert configuration["n"] == study.N
        assert configuration["nodes"] == 2
        assert len(configuration["learners"].split(" | ")) == learners
        assert "default-library" not in configuration["learners"]
        # Plan 5.13 admits the subset and control verdicts only on an influence-curve
        # interval; a plug-in-only spread would leave every row descriptive.
        assert configuration["inference"] == "influence_curve"

    def test_the_structural_cell_records_glm_fingerprints_for_every_learner(self) -> None:
        """The registry scope of SN1 names these three fingerprints (X82 SD rev 2, note 2)."""
        frame = study.draw_law("end_of_study", study.N, seed=6)
        result = study.fit_cell("SN1", frame, seed=7)
        configuration = study.trial_configuration(result, study.CELLS["SN1"].estimand)
        assert configuration["learners"].split(" | ") == [
            "GLM(family='gaussian')",
            "GLM()",
            "GLM(family='gaussian')",
        ]
        assert configuration["estimand_kind"] == "ate_regimen"
        assert configuration["treatment_kind"] == "discrete"
        assert configuration["arms"] == 2

    def test_the_structural_control_contrast_weights_each_stratum_by_its_variance(self) -> None:
        """SN1's g-estimation weights ``W2`` by ``g (1 - g)``; the static cells weight by 1/2."""
        means = study.a1_posterior_means()
        weights = {w: study.a1_propensity(w) * (1 - study.a1_propensity(w)) for w in (0, 1)}
        gaps = {w: means[(w, 1)] - means[(w, 0)] for w in (0, 1)}
        expected = (
            study.CONTROL_DELTA * sum(weights[w] * gaps[w] for w in (0, 1)) / sum(weights.values())
        )
        assert study.confounded_contrast("SN1") == pytest.approx(expected, rel=1e-14)
        assert study.confounded_contrast("SN1") != study.confounded_contrast("S1")

    @staticmethod
    def _rows(configurations: Mapping[str, Sequence[Mapping[str, object]]]) -> pd.DataFrame:
        records = [
            {"cell": f"{name}__placebo", "configuration": json.dumps(fit, sort_keys=True)}
            for name, fits in configurations.items()
            for fit in fits
        ]
        return pd.DataFrame(records)

    def test_one_configuration_per_cell_is_recorded_with_its_event_range(self) -> None:
        fits = {
            name: [
                {"method": "LTMLE", "min_event_count": 300 if name == "L1" else None},
                {"method": "LTMLE", "min_event_count": 280 if name == "L1" else None},
            ]
            for name in study.CELLS
        }
        table = properties.cell_configurations(self._rows(fits))
        assert list(table["cell"]) == list(study.CELLS)
        assert table.loc[table["cell"] == "L1", "min_event_count_min"].item() == 280
        assert table.loc[table["cell"] == "S1", "min_event_count_min"].isna().all()

    def test_a_cell_whose_configuration_varies_stops_the_run(self) -> None:
        fits = {name: [{"method": "SDR", "min_event_count": None}] for name in study.CELLS}
        fits["B3"].append({"method": "LTMLE", "min_event_count": None})
        with pytest.raises(RuntimeError, match="cell B3 fitted 2 configurations"):
            properties.cell_configurations(self._rows(fits))


class TestMutations:
    def test_the_stale_patch_keeps_the_declared_share_and_the_mask(self) -> None:
        frame = study.draw_survival(2_000, seed=8)
        result = study.fit_cell("L1", frame, seed=9)
        data = result.data
        observed = np.asarray(data.treatment, dtype=float)
        kernel = MarkovKernel.fit(data)
        original = MarkovKernel.placebo
        with study.stale_rows(0.4):
            placebo = kernel.placebo(data, np.random.default_rng(0))
        assert MarkovKernel.placebo is original
        np.testing.assert_array_equal(np.isnan(placebo), np.isnan(observed))
        same = np.all((placebo == observed) | np.isnan(observed), axis=1)
        # The kept rows agree everywhere; the drawn rest agree by chance at the kernel's rate.
        assert same.mean() >= 0.4
        fresh = kernel.placebo(data, np.random.default_rng(0))
        assert np.all((fresh == observed) | np.isnan(observed), axis=1).mean() < same.mean()

    def test_the_rescaled_subset_reading_is_the_reading_at_the_rescaled_se(self) -> None:
        from cleverly.validation._readings import subset_reading

        rng = np.random.default_rng(3)
        values = rng.normal(scale=0.02, size=study.REFITS)
        original = subset_reading(values.tolist(), std_error=0.03, n=1000, m=700)
        doubled = subset_reading(values.tolist(), std_error=0.06, n=1000, m=700)
        mutated = study.rescaled_subset(original.rho, 2.0)
        assert mutated.statistic == pytest.approx(doubled.rho, rel=1e-12)
        assert mutated.pvalue == pytest.approx(doubled.pvalue, rel=1e-10)
