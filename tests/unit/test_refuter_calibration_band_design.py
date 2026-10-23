"""The pre-run design test of ``refuter-calibration-band`` (``X82-J-plan.md`` section 5).

The plan asks four things of the declaration before it is committed.

=========================================  ==========================================================
check                                      where
=========================================  ==========================================================
the bands make a plain permutation leave   :class:`TestBands`
the band on a large share of rows
the kernel positions differ from the       :class:`TestBands`
dose space (LJ1)
the power cell's shift reaches 0.850       :class:`TestDefectSizing`, from the sizing probe
placebo SD                                 ``tests/diagnostics/x82_sj_sizing/``
the declared classifier's fit on the       :class:`TestClassifier` (PJ1, exact) and
placebo law (review M4)                    :class:`TestDefectSizing` (the probe's null bias, both)
=========================================  ==========================================================

M4 for LJ1 is the probe's ``t`` distribution at the declared configuration. The kernel draw has
no closed-form placebo ratio, and the classifier's misfit matters only through the ``t``.

The count rules are checked against their operating characteristics. The size cutoff is 20 of
200 rejections, after design review S1: :func:`TestDefectSizing.test_the_probe_t_spread` reads
PJ1's overdispersed ``t``.
"""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.stats import truncnorm

from cleverly.validation._band_placebo import PanelBands, PointBands, placebo_bands
from tests.diagnostics.x82_sj_sizing import run as sizing
from tests.studies import refuter_calibration_band as study
from tests.studies import refuter_calibration_band_properties as properties

#: The placebo shift, in placebo SD, that gives power 0.95 at ``R = 20`` (plan 8.2).
PLACEBO_SHIFT_FLOOR = 0.850
#: The Monte Carlo SD of one probe trial's ratio, about ``1 / sqrt(R)``.
TRIAL_SD = 1.0 / math.sqrt(study.REFITS)
#: Each probe mean must clear its floor by this many Monte Carlo SDs of the mean.
MARGIN_SDS = 3.0
#: The largest null bias, in placebo SD, the declaration accepts. A bias ``b`` moves the size
#: of the ``t`` rule to about 0.075 at ``b = 0.10`` (:meth:`TestDefectSizing.test_the_null_bias_bound_keeps_the_size_near_its_level`).
NULL_BIAS_BOUND = 0.10
#: The smallest share of rows a plain permutation must put outside their band.
OUTSIDE_FLOOR = 0.25

READING = pd.read_csv(sizing.HERE / "reading.csv", float_precision="round_trip")


def _probe(cell: str, quantity: str) -> pd.DataFrame:
    rows = READING[(READING["cell"] == cell) & (READING["quantity"] == quantity)]
    assert len(rows) == sizing.TRIALS[cell]
    assert set(rows["n"]) == {study.N}
    assert set(rows["refits"]) == {study.REFITS}
    return rows


def _probe_mean(cell: str, quantity: str) -> tuple[float, float]:
    """The probe's mean ratio of one quantity, and the Monte Carlo SD of that mean.

    The SD of one trial is the larger of its nominal value and the spread of the trials.
    """
    rows = _probe(cell, quantity)
    spread = float(rows["shift"].std(ddof=1))
    return float(rows["shift"].mean()), max(TRIAL_SD, spread) / math.sqrt(len(rows))


class TestLaw:
    def test_the_point_truth_agrees_with_a_large_draw(self) -> None:
        frame = study.draw_point(400_000, seed=1)
        upper = study.point_upper(frame)
        moved = frame["A"].to_numpy() + study.P_SHIFT < upper
        assert study.P_SHIFT * moved.mean() == pytest.approx(study.point_truth(), abs=0.002)

    @pytest.mark.parametrize("law", ["point", "panel"])
    def test_every_dose_lies_in_its_declared_band(self, law: str) -> None:
        frame = study.draw_law(law, 20_000, seed=2)
        if law == "point":
            bands = [(frame["A"], study.point_lower(frame), study.point_upper(frame))]
        else:
            bands = [
                (frame[f"A{t}"], study.panel_lower(frame, t), study.panel_upper(frame, t))
                for t in range(1, study.NODES + 1)
            ]
        for dose, low, high in bands:
            assert np.all((low <= dose) & (dose <= high))

    def test_the_panel_band_reads_the_engagement_and_the_earlier_price(self) -> None:
        frame = study.draw_panel(20_000, seed=3)
        for t in (2, 3):
            low = study.panel_lower(frame, t)
            width = study.panel_width(frame, t)
            assert np.corrcoef(low, frame[f"A{t - 1}"])[0, 1] > 0.5
            assert set(np.unique(width)) == {study.L_WIDTH[0], sum(study.L_WIDTH)}

    def test_a_point_draw_outside_the_declared_bounds_is_refused(self) -> None:
        with pytest.raises(RuntimeError, match="q_bounds"):
            original = study.P_Q_BOUNDS
            try:
                study.P_Q_BOUNDS = (-1.0, 1.0)
                study.draw_point(1_000, seed=4)
            finally:
                study.P_Q_BOUNDS = original


class TestBands:
    def test_a_plain_permutation_leaves_the_point_band_on_a_large_share(self) -> None:
        frame = study.draw_point(study.N, seed=5)
        dose = frame["A"].to_numpy()
        moved = dose[np.random.default_rng(0).permutation(dose.size)]
        outside = (moved < study.point_lower(frame)) | (moved > study.point_upper(frame))
        assert outside.mean() >= OUTSIDE_FLOOR

    def test_a_plain_permutation_leaves_every_panel_band_on_a_large_share(self) -> None:
        frame = study.draw_panel(study.N, seed=6)
        order = np.random.default_rng(1).permutation(study.N)
        for t in range(1, study.NODES + 1):
            moved = frame[f"A{t}"].to_numpy()[order]
            low, high = study.panel_lower(frame, t), study.panel_upper(frame, t)
            assert ((moved < low) | (moved > high)).mean() >= OUTSIDE_FLOOR

    def test_the_fits_take_the_band_rank_draw(self) -> None:
        point = study.fit_cell("PJ1", study.draw_point(study.N, seed=7), seed=8)
        panel = study.fit_cell("LJ1", study.draw_panel(study.N, seed=9), seed=10)
        assert isinstance(placebo_bands(point), PointBands)
        bands = placebo_bands(panel)
        assert isinstance(bands, PanelBands)
        assert bands.banded == (True,) * study.NODES

    def test_the_kernel_runs_on_positions_not_on_doses(self) -> None:
        frame = study.draw_panel(study.N, seed=11)
        result = study.fit_cell("LJ1", frame, seed=12)
        bands = placebo_bands(result)
        assert isinstance(bands, PanelBands)
        positions = bands.kernel.trajectories
        doses = np.asarray(result.data.treatment, dtype=float)
        assert np.nanmin(positions) >= 0.0
        assert np.nanmax(positions) <= 1.0
        assert np.nanmin(doses) > 1.0
        for t in range(1, study.NODES + 1):
            low = study.panel_lower(frame, t)
            width = study.panel_width(frame, t)
            expected = (frame[f"A{t}"].to_numpy() - low) / width
            np.testing.assert_allclose(positions[:, t - 1], expected, rtol=0, atol=1e-12)


def _marginal_position_density(u: np.ndarray) -> np.ndarray:
    """``f(u)``, law P's position density with ``W2`` integrated out (Gauss-Legendre)."""
    nodes, weights = np.polynomial.legendre.leggauss(64)
    mean = study.P_POSITION[0] + study.P_POSITION[1] * nodes
    sd = study.P_POSITION_SD
    density = truncnorm.pdf(u[:, None], (0.0 - mean) / sd, (1.0 - mean) / sd, loc=mean, scale=sd)
    return np.asarray(density @ (weights / 2.0), dtype=float)


def _placebo_ratio(dose: np.ndarray, low: np.ndarray, width: np.ndarray) -> np.ndarray:
    """The exact ratio of the policy on the placebo law of PJ1.

    The band-rank placebo draws the position from its marginal law ``f``, free of ``W``. So
    ``p'(a | W) = f((a - L) / w) / w``. The policy moves a dose below the knee ``U - 0.5`` up by
    0.5 and keeps the rest, so the policy dose has the density ``p'(b - 0.5)`` from the moved
    piece, plus ``p'(b)`` at and above the knee.
    """
    shift = study.P_SHIFT
    position = (dose - low) / width
    natural = _marginal_position_density(position) / width
    below = position - shift / width
    moved = np.where(below >= 0.0, _marginal_position_density(np.clip(below, 0.0, 1.0)), 0.0)
    moved = moved / width
    kept = np.where(dose >= low + width - shift, natural, 0.0)
    return np.asarray((moved + kept) / natural, dtype=float)


class TestClassifier:
    """Review M4 on PJ1: the declared classifier's population fit on the placebo law."""

    SIZE = 100_000

    @pytest.fixture(scope="class")
    def placebo(self) -> dict[str, np.ndarray]:
        frame = study.draw_point(self.SIZE, seed=13)
        low = study.point_lower(frame)
        width = study.point_width(frame)
        position = (frame["A"].to_numpy() - low) / width
        rng = np.random.default_rng(14)
        dose = low + position[rng.permutation(self.SIZE)] * width
        upper = low + width
        policy = np.where(dose + study.P_SHIFT < upper, dose + study.P_SHIFT, dose)
        covariates = frame[["W1", "W2"]].to_numpy()
        stacked = np.vstack(
            [np.column_stack([dose, covariates]), np.column_stack([policy, covariates])]
        )
        label = np.r_[np.zeros(self.SIZE), np.ones(self.SIZE)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            classifier = study.QuadraticLogistic("point").fit(stacked, label)
        p = classifier.predict_proba(np.column_stack([dose, covariates]))[:, 1]
        # The mean outcome given W on law P, and the population OLS of Y on (1, A', W) on the
        # placebo law, which the declared outcome learner converges to.
        frame_w = {"W1": covariates[:, 0], "W2": covariates[:, 1]}
        mean = study.P_POSITION[0] + study.P_POSITION[1] * covariates[:, 1]
        sd = study.P_POSITION_SD
        given_w = truncnorm.mean((0.0 - mean) / sd, (1.0 - mean) / sd, loc=mean, scale=sd)
        mu = (
            study.P_EFFECT * (study.point_lower(frame_w) + given_w * width)
            + study.P_OUTCOME[0] * covariates[:, 0]
            + study.P_OUTCOME[1] * covariates[:, 1]
        )
        design = np.column_stack([np.ones(self.SIZE), dose, covariates])
        q_star = design @ np.linalg.lstsq(design, mu, rcond=None)[0]
        return {
            "estimated": p / (1.0 - p),
            "exact": _placebo_ratio(dose, low, width),
            "mu": mu,
            "q_star": q_star,
        }

    def test_the_exact_ratio_has_mean_one(self, placebo: dict[str, np.ndarray]) -> None:
        assert placebo["exact"].mean() == pytest.approx(1.0, abs=0.01)

    def test_the_declared_classifier_follows_the_band_on_the_placebo_law(
        self, placebo: dict[str, np.ndarray]
    ) -> None:
        error = placebo["estimated"] - placebo["exact"]
        spread = placebo["exact"] - 1.0
        # The relative L2 error of the fitted ratio.
        assert math.sqrt(np.mean(error**2) / np.mean(spread**2)) <= CLASSIFIER_ERROR_BOUND

    def test_the_double_robust_remainder_is_small_in_placebo_sd(
        self, placebo: dict[str, np.ndarray]
    ) -> None:
        # The outcome model is misspecified on the placebo law (it omits W1 W2). The targeted
        # estimate then carries E[(r_hat - r)(mu - Q*)], which the probe's placebo SD scales.
        remainder = float(
            np.mean((placebo["estimated"] - placebo["exact"]) * (placebo["mu"] - placebo["q_star"]))
        )
        sd = float(_probe("PJ1", "fresh")["sd"].mean())
        assert abs(remainder) / sd <= NULL_BIAS_BOUND / 2.0


#: The largest relative L2 error of the declared classifier's ratio on PJ1's placebo law.
CLASSIFIER_ERROR_BOUND = 0.25


class TestDefectSizing:
    @pytest.mark.parametrize("cell", list(study.CELLS))
    def test_the_stale_share_moves_the_placebo_by_0_85_sd(self, cell: str) -> None:
        mean, sd = _probe_mean(cell, "stale")
        assert abs(mean) >= PLACEBO_SHIFT_FLOOR + MARGIN_SDS * sd

    @pytest.mark.parametrize("cell", list(study.CELLS))
    def test_the_band_rank_placebo_is_unbiased_at_the_declared_n(self, cell: str) -> None:
        mean, sd = _probe_mean(cell, "fresh")
        # The bias lies below the bound at two Monte Carlo SDs of the probe mean.
        assert abs(mean) + 2.0 * sd <= NULL_BIAS_BOUND

    @pytest.mark.parametrize("cell", list(study.CELLS))
    def test_the_probe_null_rejects_near_the_level(self, cell: str) -> None:
        rows = _probe(cell, "fresh")
        # The probe's rejection count must be one the size rule would pass at 200 trials.
        assert rows["rejected"].mean() <= study.SIZE_MAX_REJECTIONS / study.SIZE_TRIALS

    def test_the_probe_t_spread(self) -> None:
        # PJ1's refits share one sample's band-induced remainder, so its t is overdispersed
        # against t_19 (SD 1.057). The size cutoff of 20 rejections absorbs a size near 0.065.
        null = math.sqrt(19.0 / 17.0)
        pj1 = float(_probe("PJ1", "fresh")["statistic"].std(ddof=1))
        lj1 = float(_probe("LJ1", "fresh")["statistic"].std(ddof=1))
        assert null == pytest.approx(1.057, abs=5e-4)
        assert pj1 == pytest.approx(1.128, abs=5e-4)
        assert lj1 < null
        scale = pj1 / null
        critical = stats.t.ppf(1.0 - study.LEVEL / 2.0, 19)
        size = 2.0 * stats.t.sf(critical / scale, 19)
        assert 0.06 < size < 0.07

    def test_the_null_bias_bound_keeps_the_size_near_its_level(self) -> None:
        r = study.REFITS
        noncentrality = NULL_BIAS_BOUND * math.sqrt(r)
        critical = stats.t.ppf(1.0 - study.LEVEL / 2.0, r - 1)
        size = stats.nct.sf(critical, r - 1, noncentrality) + stats.nct.cdf(
            -critical, r - 1, noncentrality
        )
        assert size <= 0.076

    def test_the_placebo_floor_gives_power_0_95_at_the_default_refits(self) -> None:
        r = study.REFITS
        noncentrality = PLACEBO_SHIFT_FLOOR * math.sqrt(r)
        critical = stats.t.ppf(1.0 - study.LEVEL / 2.0, r - 1)
        power = stats.nct.sf(critical, r - 1, noncentrality) + stats.nct.cdf(
            -critical, r - 1, noncentrality
        )
        assert power >= 0.949

    def test_the_probe_ran_at_the_declared_defect_sizes(self) -> None:
        assert set(READING["cell"]) == set(study.CELLS)
        for cell in study.CELLS:
            rows = READING[READING["cell"] == cell]
            assert set(rows["stale_share"]) == {study.STALE_SHARE[cell]}


class TestCountRules:
    def test_the_interval_thresholds_reproduce_the_count_rules(self) -> None:
        assert properties.count_rule_agrees()

    @pytest.mark.parametrize(
        ("size", "expected"),
        [(0.05, 0.9988), (0.065, 0.9790), (0.07, 0.9582), (0.10, 0.5592), (0.15, 0.0255)],
    )
    def test_the_size_rule_operating_characteristic(self, size: float, expected: float) -> None:
        passed = stats.binom.cdf(study.SIZE_MAX_REJECTIONS, study.SIZE_TRIALS, size)
        assert passed == pytest.approx(expected, abs=5e-4)

    def test_both_size_readings_pass_together_with_probability_0_96(self) -> None:
        # PJ1 at the probe's size near 0.065 and LJ1 at the level (design review S1).
        pj1 = stats.binom.cdf(study.SIZE_MAX_REJECTIONS, study.SIZE_TRIALS, 0.065)
        lj1 = stats.binom.cdf(study.SIZE_MAX_REJECTIONS, study.SIZE_TRIALS, study.LEVEL)
        assert pj1 * lj1 >= 0.975

    @pytest.mark.parametrize(
        ("power", "expected"), [(0.95, 1.0), (0.80, 0.8139), (0.70, 0.2229), (0.50, 1.53e-4)]
    )
    def test_the_power_rule_operating_characteristic(self, power: float, expected: float) -> None:
        passed = stats.binom.sf(study.POWER_MIN_DETECTIONS - 1, study.POWER_TRIALS, power)
        assert passed == pytest.approx(expected, abs=5e-4)


class TestCells:
    def test_the_cells_are_the_plan_cells(self) -> None:
        assert set(study.CELLS) == {"PJ1", "LJ1"}
        assert len(study.PROPERTY_CELLS[study.SIZE_FAMILY]) == 2
        assert len(study.PROPERTY_CELLS[study.POWER_FAMILY]) == 2
        assert {cell.placebo_draw for cell in study.CELLS.values()} == {
            "band_permute",
            "band_markov",
        }

    def test_every_power_trial_is_a_size_trial(self) -> None:
        assert study.POWER_TRIALS <= study.SIZE_TRIALS

    def test_the_cell_configurations_are_a_committed_artifact(self) -> None:
        assert study.STUDY.property_artifacts == (properties.CELLS_FILE,)


class TestConfiguration:
    @pytest.mark.parametrize(("cell", "method"), [("PJ1", "TMLE"), ("LJ1", "LTMLE")])
    def test_a_trial_records_the_fitted_configuration(self, cell: str, method: str) -> None:
        spec = study.CELLS[cell]
        result = study.fit_cell(cell, study.draw_law(spec.law, study.N, seed=15), seed=16)
        configuration = study.trial_configuration(result, spec)
        assert configuration["method"] == method
        assert configuration["treatment_kind"] == "continuous"
        assert configuration["cross_fitting"] == f"folds={study.N_FOLDS}"
        assert configuration["replicates"] == study.REFITS
        assert configuration["placebo_draw"] == spec.placebo_draw
        assert configuration["n"] == study.N
        assert "QuadraticLogistic" in configuration["learners"]

    def test_the_refute_report_records_the_cell_draw(self) -> None:
        spec = study.CELLS["PJ1"]
        result = study.fit_cell("PJ1", study.draw_point(study.N, seed=17), seed=18)
        test = study._placebo(result, spec, seed=19)
        assert "keeps each unit's declared band" in test.detail

    @staticmethod
    def _rows(configurations: dict[str, list[dict[str, object]]]) -> pd.DataFrame:
        records = [
            {"cell": f"{name}__placebo", "configuration": json.dumps(fit, sort_keys=True)}
            for name, fits in configurations.items()
            for fit in fits
        ]
        return pd.DataFrame(records)

    def test_a_cell_whose_configuration_varies_stops_the_run(self) -> None:
        fits = {name: [{"method": "TMLE", "min_event_count": None}] for name in study.CELLS}
        fits["LJ1"].append({"method": "SDR", "min_event_count": None})
        with pytest.raises(RuntimeError, match="cell LJ1 fitted 2 configurations"):
            properties.cell_configurations(self._rows(fits))


class TestMutations:
    def test_the_point_stale_patch_keeps_the_declared_share_and_restores(self) -> None:
        result = study.fit_cell("PJ1", study.draw_point(study.N, seed=20), seed=21)
        bands = placebo_bands(result)
        assert isinstance(bands, PointBands)
        original = PointBands.draw
        observed = np.asarray(result.data.treatment, dtype=float)
        with study.stale_rows(0.3):
            placebo = bands.draw(result.data, np.random.default_rng(0), None)
        assert PointBands.draw is original
        assert np.mean(placebo == observed) >= 0.3
        fresh = bands.draw(result.data, np.random.default_rng(0), None)
        assert np.mean(fresh == observed) < 0.01

    def test_the_panel_stale_patch_keeps_whole_trajectories_and_restores(self) -> None:
        result = study.fit_cell("LJ1", study.draw_panel(study.N, seed=22), seed=23)
        bands = placebo_bands(result)
        assert isinstance(bands, PanelBands)
        original = PanelBands.placebo
        observed = np.asarray(result.data.treatment, dtype=float)
        with study.stale_rows(0.3):
            doses, drawn = bands.placebo(result.data, np.random.default_rng(0))
        assert PanelBands.placebo is original
        kept = np.all(doses == observed, axis=1)
        assert kept.mean() >= 0.3
        np.testing.assert_array_equal(drawn[kept], bands.kernel.trajectories[kept])
