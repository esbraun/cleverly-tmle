"""The pre-run design test of ``refuter-calibration-continuous-survival`` (X101 plan rev 2, section 5).

The power law and the stale-row defect are sized here from the exact law, and checked against
the committed pilot (``tests/studies/refuter_calibration_continuous_survival_pilot.csv.gz``,
on its own seed stream). The predictive family pass of the eight size cells is computed from the
pilot's per-cell rejections and the correlation of its readings. The shared-refit cache, the
stale-row patch and the configuration record are checked on a small fit.
"""

from __future__ import annotations

import json
import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from cleverly.survival import ContinuousSurvivalTMLE, CoxHazard
from cleverly.survival.engine import SURVIVAL_FLOOR
from cleverly.validation import _continuous_survival_refute as adapter
from cleverly.validation import refute
from tests.studies import continuous_survival_law as law
from tests.studies import refuter_calibration_continuous_survival as study
from tests.studies import refuter_calibration_continuous_survival_pilot as pilot
from tests.studies import refuter_calibration_continuous_survival_properties as properties

PILOT = pilot.read()
#: The predictive family pass the size cells must reach before the declaration (plan 5).
FAMILY_PASS_FLOOR = 0.90
#: The per-trial power the stale-row share must buy on every estimand (plan 5).
STALE_POWER_FLOOR = 0.95


def _z(*, power: bool, beta: float | None = None) -> dict[str, float]:
    """Each difference in efficiency SE at ``N``, on the power or size law, or at ``beta``."""
    coefficients = (
        (study.POWER_BETA, study.POWER_CAUSE2_ARM) if power else (law.BETA, law.CAUSE2_ARM)
    )
    chosen_beta = coefficients[0] if beta is None else beta
    cause2_arm = coefficients[1]
    truth = law.truth(study.GATED_TIMES, beta=chosen_beta, cause2_arm=cause2_arm)
    out = {}
    for key, name in study.ESTIMANDS.items():
        code, t = int(key[1]), float(key.split("_t")[1])
        sd = law.efficiency_sd(code, t, beta=chosen_beta, cause2_arm=cause2_arm)["difference"]
        out[key] = truth[name] / (sd / math.sqrt(study.N))
    return out


def _worst_treated_survival(beta: float, cause2_arm: float = law.CAUSE2_ARM) -> float:
    """The true treated survival at the last target time of a unit with X1 = 1, X3 -> 1-."""
    h1, h2 = law.rates(
        np.array(1.0), np.array(0.0), np.array(1.0 - 1e-9), 1, beta=beta, cause2_arm=cause2_arm
    )
    return float(np.exp(-(h1 + h2) * law.TIMES[-1]))


class TestLaw:
    def test_every_power_law_difference_is_at_least_four_se(self) -> None:
        assert all(abs(z) >= study.POWER_EFFECT_SE for z in _z(power=True).values())

    def test_the_size_law_differences_are_far_from_four_se(self) -> None:
        # Plan finding M2: the stale-row power is unreachable on the registered law.
        assert min(abs(z) for z in _z(power=False).values()) < 1.0

    def test_raising_beta_alone_reaches_the_survival_floor_first(self) -> None:
        """The deviation from the plan: beta alone needs 3.5, where the fit raises."""
        assert abs(_z(power=False, beta=3.25)["c2_t1.12"]) < study.POWER_EFFECT_SE
        assert abs(_z(power=False, beta=3.5)["c2_t1.12"]) >= study.POWER_EFFECT_SE
        assert _worst_treated_survival(3.5) < SURVIVAL_FLOOR
        assert _worst_treated_survival(study.POWER_BETA, study.POWER_CAUSE2_ARM) > 1e-3

    def test_the_power_law_draws_its_coefficients(self) -> None:
        base = study.draw_law("size", 50_000, seed=1)
        power = study.draw_law("power", 50_000, seed=1)
        np.testing.assert_array_equal(base["Z"], power["Z"])
        treated = base["Z"] == 1
        assert power.loc[treated, "time"].mean() < base.loc[treated, "time"].mean()
        np.testing.assert_array_equal(
            base.loc[~treated, "time"].to_numpy(), power.loc[~treated, "time"].to_numpy()
        )

    def test_the_imbalanced_cell_treats_a_quarter(self) -> None:
        frame = study.draw_law("imbalanced", 200_000, seed=2)
        assert frame["Z"].mean() == pytest.approx(study.IMBALANCED_SHARE, abs=0.004)

    def test_the_dependent_censoring_cell_censors_by_arm(self) -> None:
        frame = study.draw_law("dependent_censoring", 200_000, seed=3)
        censored = frame["event"] == 0
        assert frame.loc[frame["Z"] == 1, "event"].eq(0).mean() > censored.mean()


class TestDefectSizing:
    def test_a_fully_stale_placebo_reads_the_power_law_difference(self) -> None:
        shift, truth = study.stale_shift(1.0), study.power_truth()
        for key in study.ESTIMANDS:
            assert shift[key] == pytest.approx(truth[key], abs=1e-12)

    def test_a_fresh_placebo_has_no_shift(self) -> None:
        assert all(
            value == pytest.approx(0.0, abs=1e-15) for value in study.stale_shift(0.0).values()
        )

    def test_the_pilot_reads_the_exact_stale_shift(self) -> None:
        """The probe: the pilot's power trials centre on the exact shift at the declared share."""
        power = PILOT.loc[(PILOT["group"] == "power") & PILOT["refusal"].fillna("").eq("")]
        shift = study.stale_shift(study.STALE_SHARE)
        for key in study.ESTIMANDS:
            means = power.loc[power["estimand"] == key].drop_duplicates("trial")["placebo_mean"]
            error = float(means.std(ddof=1)) / math.sqrt(len(means))
            assert float(means.mean()) == pytest.approx(shift[key], abs=4.0 * error)
            assert abs(shift[key]) > 4.0 * error

    def test_the_stale_share_buys_per_trial_power_0_95(self) -> None:
        power = pilot.stale_power(PILOT)
        assert min(power.values()) >= STALE_POWER_FLOOR, power

    def test_the_share_is_not_oversized(self) -> None:
        # 0.30 also reaches the floor on the pilot's moments (0.987 at c2_t1.12); the declared
        # share is the one the pilot ran, and 0.10 less falls short.
        power = pilot.stale_power(PILOT, share=study.STALE_SHARE - 0.10)
        assert min(power.values()) < STALE_POWER_FLOOR, power

    def test_every_pilot_power_trial_detected_the_defect(self) -> None:
        power = PILOT.loc[PILOT["group"] == "power"]
        assert power["refusal"].fillna("").eq("").all()
        assert power["rejected"].eq(1).all()

    @pytest.mark.parametrize("factor", [0.5, 2.0])
    def test_a_halved_or_doubled_se_is_detected_with_power_0_98(self, factor: float) -> None:
        df = study.REFITS - 1
        low, high = stats.chi2.ppf([study.LEVEL / 2.0, 1.0 - study.LEVEL / 2.0], df)
        scale = 1.0 / factor**2
        power = stats.chi2.cdf(low / scale, df) + stats.chi2.sf(high / scale, df)
        assert power >= 0.98


class TestFamily:
    def test_the_size_cells_pass_together_with_probability_0_90(self) -> None:
        """Plan 5: the declaration waits for a family pass of 0.90; pending says why."""
        value = pilot.family_pass(pilot.signed_scores(PILOT))
        if study.DECLARATION_PENDING:
            assert value < FAMILY_PASS_FLOOR
        else:
            assert value >= FAMILY_PASS_FLOOR

    def test_the_family_rule_reads_0_907_at_exact_size_and_independence(self) -> None:
        rng = np.random.default_rng(4)
        scores = pd.DataFrame(rng.standard_normal((200_000, 8)))
        assert pilot.family_pass(scores, draws=2_000) == pytest.approx(0.907, abs=0.02)

    def test_the_scores_reproduce_the_recorded_rejections(self) -> None:
        scores = pilot.signed_scores(PILOT)
        size = PILOT.loc[(PILOT["group"] == "size") & PILOT["reading"].isin(study.SIZE_READINGS)]
        recorded = size.assign(cell=size["estimand"] + "__" + size["reading"]).pivot(
            index="trial", columns="cell", values="rejected"
        )
        critical = stats.norm.ppf(1.0 - study.LEVEL / 2.0)
        found = (scores.abs() > critical).astype(int)
        expected = recorded.loc[:, found.columns].astype(int)
        pd.testing.assert_frame_equal(found, expected, check_names=False)

    def test_every_pilot_refusal_is_the_survival_floor_on_a_subset_refit(self) -> None:
        """X6 PR A removes this DataError; until then the pilot's loss rate keeps it pending."""
        refused = PILOT.loc[PILOT["refusal"].fillna("").ne("")]
        assert set(refused["refusal"]) == {"DataError"}
        assert set(refused["group"]) == {"size"}
        assert refused["message"].str.contains("on subset refit").all()
        assert refused["message"].str.contains("survival reaches zero").all()
        trials = PILOT.loc[PILOT["group"] == "size", "trial"].nunique()
        share = refused["trial"].nunique() / trials
        assert share > properties.FAILURE_BOUND
        assert study.DECLARATION_PENDING


class TestCountRules:
    def test_the_interval_thresholds_reproduce_the_count_rules(self) -> None:
        assert properties.count_rule_agrees()

    def test_the_ceiling_count_at_the_declared_size(self) -> None:
        ceiling = pilot.count_ceiling(study.SIZE_TRIALS, properties.SIZE_CEILING)
        assert ceiling == study.SIZE_MAX_REJECTIONS

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
        assert len(study.PROPERTY_CELLS[study.SIZE_FAMILY]) == 8
        assert len(study.PROPERTY_CELLS[study.POWER_FAMILY]) == 12
        assert set(study.ESTIMANDS.values()) == {
            law.difference_name(cause, t) for cause in law.CAUSES.values() for t in (1.12, 2.32)
        }

    def test_every_rescaled_subset_trial_is_a_size_trial(self) -> None:
        assert study.POWER_TRIALS <= study.SIZE_TRIALS

    def test_the_streams_are_distinct(self) -> None:
        streams = (study.STREAM, study.POWER_STREAM, study.REPORTING_STREAM, study.PILOT_STREAM)
        assert len(set(streams)) == 4
        seeds = {
            study.trial_seed(stream, "sample", group, 0)
            for stream in streams
            for group in study.GROUPS
        }
        assert len(seeds) == len(streams) * len(study.GROUPS)

    def test_the_payloads_cover_the_declared_trials(self) -> None:
        keys = properties.payloads()
        assert keys.count(("size", 0)) == 1
        assert sum(group == "size" for group, _ in keys) == study.SIZE_TRIALS
        assert sum(group == "power" for group, _ in keys) == study.POWER_TRIALS
        assert len(properties._cells_of(("size", 0))) == 16
        assert len(properties._cells_of(("size", study.POWER_TRIALS))) == 8
        assert len(properties._cells_of(("power", 0))) == 4

    def test_the_committed_artifacts_include_the_reporting_cells(self) -> None:
        assert study.STUDY.property_artifacts == (
            properties.CELLS_FILE,
            properties.REPORTING_CELLS_FILE,
            properties.REPORTING_REPLICATES_FILE,
        )

    def test_the_pilot_ran_the_declared_fit_and_scope(self) -> None:
        kept = PILOT.loc[PILOT["refusal"].fillna("").eq("")]
        assert set(kept["inference"]) == {"influence_curve"}
        assert kept.loc[kept["group"] == "size", "trial"].nunique() == 57
        size = kept.loc[kept["group"] == "size"]
        assert set(size["fits"]) == {1 + 2 * study.REFITS}
        assert set(kept.loc[kept["group"] != "size", "fits"]) == {1 + study.REFITS}


# ---- the machinery, on a small fit ---------------------------------------------------------

RD = "ate_regimen[1 vs 0, {cause} @ t={t:g}]"


def _small_fit() -> Any:
    rng = np.random.default_rng(0)
    n = 300
    w = rng.normal(size=n)
    a = rng.binomial(1, 1 / (1 + np.exp(-w)))
    h1 = np.exp(-1.5 + 0.5 * w - 1.0 * a)
    h2 = np.exp(-2.0 - 0.5 * w - 0.5 * a)
    t = rng.exponential(1 / (h1 + h2))
    cause = np.where(rng.uniform(size=n) < h1 / (h1 + h2), 1, 2)
    c = rng.exponential(1 / 0.15, n)
    frame = pd.DataFrame(
        {"time": np.minimum(t, c), "event": np.where(t <= c, cause, 0), "A": a, "W": w}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ContinuousSurvivalTMLE(
            [1.0, 2.0],
            cross_fit=True,
            n_folds=3,
            hazard_library=[CoxHazard("main")],
            censoring_library=[CoxHazard("main")],
            random_state=0,
        ).fit(frame, time="time", event="event", treatment="A", baseline=["W"])


@pytest.fixture(scope="module")
def small() -> Any:
    return _small_fit()


def _refute(result: Any, estimand: str) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return refute(
            result, estimand=estimand, tests=("placebo", "subset"), n_replicates=4, random_state=9
        )


class TestSharedRefits:
    def test_four_estimands_share_one_set_of_refits_bit_for_bit(self, small: Any) -> None:
        names = [RD.format(cause=c, t=t) for c in (1, 2) for t in (1.0, 2.0)]
        alone = {name: _refute(small, name) for name in names}
        original = ContinuousSurvivalTMLE.refit
        with study.shared_refits() as log:
            shared = {name: _refute(small, name) for name in names}
        assert ContinuousSurvivalTMLE.refit is original
        # Four placebo and four subset refits, asked for four times.
        assert (log.fits, log.calls) == (8, 32)
        for name in names:
            for test in ("placebo", "subset"):
                assert shared[name][test].values == alone[name][test].values
                assert shared[name][test].statistic == alone[name][test].statistic

    def test_a_different_seed_is_a_different_refit(self, small: Any) -> None:
        name = RD.format(cause=1, t=2.0)
        with study.shared_refits() as log, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for seed in (1, 2):
                refute(small, estimand=name, tests=("placebo",), n_replicates=3, random_state=seed)
        assert log.fits == 6


class TestMutations:
    def test_the_stale_patch_keeps_the_declared_share_and_restores(self) -> None:
        treatment = np.repeat([0.0, 1.0], 500)
        original = adapter._placebo_treatment
        with study.stale_rows(0.35):
            placebo = adapter._placebo_treatment(treatment, np.random.default_rng(0), "permute")
        assert adapter._placebo_treatment is original
        assert placebo.sum() == treatment.sum()
        fresh = original(treatment, np.random.default_rng(0), "permute")
        # The kept rows agree; the permuted rest agree by chance about half the time.
        assert (placebo == treatment).mean() >= 0.35 + 0.5 * 0.65 - 0.05
        assert (fresh == treatment).mean() < (placebo == treatment).mean()

    def test_the_stale_patch_moves_the_placebo_on_a_fit(self, small: Any) -> None:
        name = RD.format(cause=1, t=2.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fresh = refute(small, estimand=name, tests=("placebo",), n_replicates=4, random_state=3)
            with study.stale_rows(0.9):
                stale = refute(
                    small, estimand=name, tests=("placebo",), n_replicates=4, random_state=3
                )
        assert fresh["placebo"].values != stale["placebo"].values

    def test_the_rescaled_subset_reading_is_the_reading_at_the_rescaled_se(self) -> None:
        from cleverly.validation._readings import subset_reading

        rng = np.random.default_rng(3)
        values = rng.normal(scale=0.02, size=study.REFITS)
        original = subset_reading(values.tolist(), std_error=0.03, n=1000, m=700)
        doubled = subset_reading(values.tolist(), std_error=0.06, n=1000, m=700)
        mutated = study.rescaled_subset(original.rho, 2.0)
        assert mutated.statistic == pytest.approx(doubled.rho, rel=1e-12)
        assert mutated.pvalue == pytest.approx(doubled.pvalue, rel=1e-10)


class TestConfiguration:
    @staticmethod
    def _rows(fits: list[dict[str, Any]]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "cell": f"{key}__placebo",
                    "configuration": json.dumps({**fit, "law": "size"}, sort_keys=True),
                }
                for key in study.ESTIMANDS
                for fit in fits
            ]
        )

    def test_one_configuration_per_estimand_with_its_ranges(self) -> None:
        fits = [
            {"method": "ContinuousSurvivalTMLE", "min_event_count": 30, "event_share": 0.5},
            {"method": "ContinuousSurvivalTMLE", "min_event_count": 25, "event_share": 0.55},
        ]
        table = properties.cell_configurations(self._rows(fits))
        assert list(table["cell"]) == list(study.ESTIMANDS)
        assert set(table["min_event_count_min"]) == {25}
        assert set(table["event_share_max"]) == {0.55}

    def test_a_configuration_that_varies_stops_the_run(self) -> None:
        fits = [
            {"method": "ContinuousSurvivalTMLE", "min_event_count": 30, "event_share": 0.5},
            {"method": "TMLE", "min_event_count": 30, "event_share": 0.5},
        ]
        with pytest.raises(RuntimeError, match="fitted 2 configurations"):
            properties.cell_configurations(self._rows(fits))

    def test_the_trial_configuration_reads_the_scope_axes(self, small: Any) -> None:
        row = study.trial_configuration(small, RD.format(cause=1, t=2.0))
        assert row["replicates"] == study.REFITS
        assert row["causes"] == 2
        assert row["estimand_kind"] == "continuous:ate_regimen"
        assert 0.0 < row["event_share"] <= 1.0
