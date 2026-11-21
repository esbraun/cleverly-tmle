"""The pre-run design test of ``refuter-sampled`` (X106 part H, comparison plan section 6).

Each defect level is sized here from the declared law, the gates' operating characteristics are
computed from the binomial (never copied), and the gate functions are run on synthetic rows.
Nothing here fits an estimator except the one clustered-law check, which draws data only.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from cleverly.validation._readings import subset_reference_sd
from tests.studies import refuter_calibration_point as point
from tests.studies import refuter_sampled as study
from tests.studies import refuter_sampled_properties as properties

LAW = point.quadrature()


def _placebo_power(shift: float) -> float:
    r = study.REFITS
    critical = stats.t.ppf(1.0 - study.LEVEL / 2.0, r - 1)
    ncp = shift * math.sqrt(r)
    return float(stats.nct.sf(critical, r - 1, ncp) + stats.nct.cdf(-critical, r - 1, ncp))


def _control_power(shift: float) -> float:
    z = stats.norm.ppf(1.0 - study.LEVEL / 2.0)
    return float(stats.norm.cdf(shift - z) + stats.norm.cdf(-shift - z))


def _subset_power(factor: float) -> float:
    df = study.REFITS - 1
    low, high = stats.chi2.ppf([study.LEVEL / 2.0, 1.0 - study.LEVEL / 2.0], df)
    return float(stats.chi2.cdf(low * factor**2, df) + stats.chi2.sf(high * factor**2, df))


class TestLaw:
    def test_the_iid_law_is_the_registered_t1_law(self) -> None:
        frame = study.draw_law(study.CELLS["I2-F"], seed=3)
        expected = point.draw_law("gaussian", 2_000, seed=3)
        pd.testing.assert_frame_equal(frame, expected)

    def test_the_clustered_law_keeps_the_ate_and_the_null_control(self) -> None:
        from sklearn.linear_model import LinearRegression

        cell = study.Cell(20_000, None, True, 1)
        frame = study.draw_law(cell, seed=4)
        assert frame["cl"].value_counts().eq(study.CLUSTER_SIZE).all()
        design = frame[["A", *study.COVARIATES]].to_numpy()
        outcome = LinearRegression().fit(design, frame["Y"]).coef_[0]
        control = LinearRegression().fit(design, frame["NC"]).coef_[0]
        assert outcome == pytest.approx(point.GAUSSIAN_EFFECT, abs=0.03)
        assert control == pytest.approx(0.0, abs=0.03)

    def test_the_clustered_fit_has_a_design_effect_of_at_least_1_4(self) -> None:
        """The nonzero witness that the C cells can see cluster handling."""
        cell = study.CELLS["C-T"]
        frame = study.draw_law(cell, seed=7)
        clustered = study.fit_cell(cell, frame, seed=8)
        rows = study.fit_cell(study.Cell(cell.rows, None, False, 1), frame, seed=8)
        ratio = (clustered["ate"].plugin_std_error / rows["ate"].plugin_std_error) ** 2
        assert ratio >= 1.4

    def test_the_clustered_law_correlates_members(self) -> None:
        frame = study.draw_law(study.Cell(4_000, None, True, 1), seed=5)
        residual = frame["NC"] - point.outcome_index(frame[list(study.COVARIATES)].to_numpy())
        means = residual.groupby(frame["cl"]).mean()
        # Var of a cluster mean of five: 0.25 + 1 / 5 under the declared law.
        assert means.var() == pytest.approx(study.CLUSTER_SD**2 + 1 / study.CLUSTER_SIZE, rel=0.1)

    def test_the_rebuilt_control_at_the_strong_level_is_the_drawn_control(self) -> None:
        shift = study.rebuilt_control_shift(study.STRONG_CONTROL_DELTA, 2_000, LAW)
        expected = point.control_shift("gaussian", study.STRONG_CONTROL_DELTA, n=2_000, law=LAW)
        assert shift == pytest.approx(expected, rel=1e-12)

    def test_the_weak_control_is_the_strong_one_scaled(self) -> None:
        frame = point.draw_law("gaussian", 10, seed=6)
        rebuilt = study.confounded_control(frame, study.STRONG_CONTROL_DELTA)
        np.testing.assert_allclose(rebuilt, frame["NCU"].to_numpy(), rtol=1e-12)


class TestDefectSizing:
    def test_the_strong_defects_have_power_0_95_at_m_2000(self) -> None:
        placebo = point.placebo_shift("gaussian", study.STRONG_STALE_SHARE, n=2_000, law=LAW)
        control = point.control_shift("gaussian", study.STRONG_CONTROL_DELTA, n=2_000, law=LAW)
        assert _placebo_power(placebo) >= 0.95
        assert _control_power(control) >= 0.95
        assert min(_subset_power(f) for f in study.STRONG_SE_FACTORS) >= 0.98

    def test_the_weak_defects_have_power_near_one_half_at_m_2000(self) -> None:
        placebo = point.placebo_shift("gaussian", study.WEAK_STALE_SHARE, n=2_000, law=LAW)
        assert 0.4 <= _placebo_power(placebo) <= 0.6
        control = study.rebuilt_control_shift(study.WEAK_CONTROL_DELTA, 2_000, LAW)
        assert 0.4 <= _control_power(control) <= 0.6
        for factor in study.WEAK_SE_FACTORS:
            assert 0.4 <= _subset_power(factor) <= 0.6

    def test_the_weak_level_shows_the_cost_of_sampling(self) -> None:
        # At the full n of a P cell the same defect is near certain; at m = 1,000 it is weak.
        for n, low, high in ((1_000, 0.2, 0.4), (20_000, 0.99, 1.0)):
            shift = point.placebo_shift("gaussian", study.WEAK_STALE_SHARE, n=n, law=LAW)
            assert low <= _placebo_power(shift) <= high

    def test_the_mutation_rejects_most_null_trials_at_one_tenth(self) -> None:
        # The full fit's SE is about SE_m sqrt(m / n); reading the sampled refits against it
        # scales rho by the ratio of the two subset reference SDs.
        for m, n in ((2_000, 20_000), (1_000, 10_000), (400, 4_000)):
            k = min(m, max(20, math.floor(study.SUBSET_FRACTION * m)))
            own = subset_reference_sd(1.0, m, k)
            sliced = subset_reference_sd(math.sqrt(m / n), n, k)
            assert _subset_power(sliced / own) >= 0.8


class TestGates:
    def test_the_interval_thresholds_reproduce_the_count_rules(self) -> None:
        assert properties.count_rule_agrees()

    def test_the_absolute_size_rule_operating_characteristic(self) -> None:
        trials, k = study.SIZE_TRIALS, study.SIZE_MAX_REJECTIONS
        assert stats.binom.cdf(k, trials, 0.05) >= 0.999
        assert stats.binom.cdf(k, trials, 0.10) <= 0.002

    def test_the_power_rule_operating_characteristic(self) -> None:
        trials, k = study.POWER_TRIALS, study.POWER_MIN_DETECTIONS
        assert stats.binom.sf(k - 1, trials, 0.90) >= 0.9989

    @staticmethod
    def _relative_pass(p: float, q: float) -> float:
        """The exact probability that the relative gate passes at rates ``p`` and ``q``."""
        trials = study.SIZE_TRIALS
        counts = np.arange(trials + 1)
        own = stats.binom.pmf(counts, trials, p)
        partner = stats.binom.pmf(counts, trials, q)
        total = 0.0
        for a in counts[own > 1e-12]:
            for b in counts[partner > 1e-12]:
                x = np.zeros(trials)
                x[:a] = 1.0
                y = np.zeros(trials)
                y[:b] = 1.0
                if properties.relative_passes(x, y):
                    total += own[a] * partner[b]
        return total

    def test_the_relative_rule_operating_characteristic(self) -> None:
        assert self._relative_pass(0.05, 0.05) >= 0.999
        assert self._relative_pass(0.06, 0.06) >= 0.997
        assert self._relative_pass(0.12, 0.05) <= 0.01

    def test_the_family_false_red_is_at_most_0_05(self) -> None:
        """The Bonferroni sum over the gated readings, which needs no independence."""
        size = 1.0 - stats.binom.cdf(study.SIZE_MAX_REJECTIONS, study.SIZE_TRIALS, 0.05)
        relative = 1.0 - self._relative_pass(0.05, 0.05)
        power = 1.0 - stats.binom.sf(study.POWER_MIN_DETECTIONS - 1, study.POWER_TRIALS, 0.90)
        sizes = len(study.PROPERTY_CELLS[study.SIZE_FAMILY])
        relatives = len(study.PROPERTY_CELLS[study.RELATIVE_FAMILY])
        powers = len(study.PROPERTY_CELLS[study.POWER_FAMILY])
        assert sizes * size + relatives * relative + powers * power <= 0.03

    def test_the_bound_is_conditional_on_exact_size(self) -> None:
        # The declaration's stated fragility: at size 0.06 the nine absolute readings alone
        # turn red with probability about 0.30.
        green = stats.binom.cdf(study.SIZE_MAX_REJECTIONS, study.SIZE_TRIALS, 0.06) ** 9
        assert 1.0 - green == pytest.approx(0.30, abs=0.02)


class TestCells:
    def test_the_cells_are_the_plan_cells(self) -> None:
        assert set(study.CELLS) == {
            "I2-F", "I2-H", "I2-T", "I1-F", "I1-T", "C-F", "C-H", "C-T", "P-4k", "P-20k"
        }  # fmt: skip
        counts = {family: len(cells) for family, cells in study.PROPERTY_CELLS.items()}
        assert counts == {
            study.SIZE_FAMILY: 9,
            study.RELATIVE_FAMILY: 15,
            study.POWER_FAMILY: 12,
            study.MUTATION_FAMILY: 3,
        }

    def test_each_sampled_cell_is_read_against_its_full_cell_at_n_equal_m(self) -> None:
        for name in study.RELATIVE_CELLS:
            cell = study.CELLS[name]
            partner = study.CELLS[cell.partner]
            assert partner.m is None and partner.n == cell.m
            assert partner.clustered == cell.clustered

    def test_a_full_run_writes_exactly_the_declared_gated_cells(self) -> None:
        written = properties.written_cells()
        for family, cells in study.PROPERTY_CELLS.items():
            assert {cell for name, cell in written if name == family} >= set(cells)
        partners = {cell for family, cell in written if cell.endswith(properties.PARTNER)}
        assert len(partners) == 15

    def test_the_streams_are_distinct(self) -> None:
        seeds = {
            study.trial_seed(f"{stream}_sample", name, trial)
            for stream in properties.STREAMS
            for name in study.CELLS
            for trial in range(3)
        }
        assert len(seeds) == len(properties.STREAMS) * len(study.CELLS) * 3


def _rows(rates: dict[tuple[str, str], float], trials: int = 1_000) -> pd.DataFrame:
    """Synthetic rows with exactly ``rate * trials`` rejections in each listed cell."""
    records = []
    for family, cells in study.PROPERTY_CELLS.items():
        for cell in cells:
            count = study.POWER_TRIALS if family == study.POWER_FAMILY else trials
            rate = rates.get((family, cell), 0.95 if family == study.POWER_FAMILY else 0.05)
            rejected = np.zeros(count, dtype=int)
            rejected[: round(rate * count)] = 1
            for trial in range(count):
                records.append((family, cell, trial, count, int(rejected[trial])))
    for name in study.RELATIVE_CELLS:
        for reading in study.READINGS:
            cell = f"{name}__{reading}{properties.PARTNER}"
            rejected = np.zeros(trials, dtype=int)
            rejected[: round(rates.get(("partner", cell), 0.05) * trials)] = 1
            for trial in range(trials):
                records.append((study.RELATIVE_FAMILY, cell, trial, trials, int(rejected[trial])))
    frame = pd.DataFrame(
        records, columns=["property", "cell", "replicate", "requested_replicates", "rejected"]
    )
    frame["role"] = [
        properties._role(family, cell)
        for family, cell in zip(frame["property"], frame["cell"], strict=True)
    ]
    frame["n"] = 2_000
    frame["failed_replicates"] = 0
    frame["truth"] = 0.0
    frame["estimate"] = 0.0
    frame["std_error"] = 1.0
    frame["covered"] = 1 - frame["rejected"]
    return frame


class TestVerdicts:
    @staticmethod
    def _passed(summary: pd.DataFrame, family: str, cell: str) -> bool:
        row = summary.loc[(summary["property"] == family) & (summary["cell"] == cell)]
        return bool(row["passed"].iloc[0])

    def test_a_calibrated_sampled_path_and_a_detected_mutation_pass_every_gate(self) -> None:
        mutation = {
            (study.MUTATION_FAMILY, cell): 0.6
            for cell in study.PROPERTY_CELLS[study.MUTATION_FAMILY]
        }
        summary = properties.summarize_properties(_rows(mutation))
        gated = summary.loc[summary["role"] != "diagnostic"]
        assert gated["passed"].all()

    def test_an_oversized_sampled_cell_fails_its_size_and_relative_gates(self) -> None:
        rates = {
            (study.SIZE_FAMILY, "I2-T__placebo"): 0.10,
            (study.RELATIVE_FAMILY, "I2-T__placebo"): 0.10,
        }
        summary = properties.summarize_properties(_rows(rates))
        assert not self._passed(summary, study.SIZE_FAMILY, "I2-T__placebo")
        assert not self._passed(summary, study.RELATIVE_FAMILY, "I2-T__placebo")
        assert self._passed(summary, study.RELATIVE_FAMILY, "I2-H__placebo")

    def test_a_mutation_that_reads_like_the_partner_fails(self) -> None:
        summary = properties.summarize_properties(_rows({}))
        for cell in study.PROPERTY_CELLS[study.MUTATION_FAMILY]:
            assert not self._passed(summary, study.MUTATION_FAMILY, cell)

    def test_a_weak_power_cell_fails(self) -> None:
        rates = {(study.POWER_FAMILY, "I2-H__placebo_stale_rows"): 0.6}
        summary = properties.summarize_properties(_rows(rates))
        assert not self._passed(summary, study.POWER_FAMILY, "I2-H__placebo_stale_rows")


class TestRules:
    def test_each_rule_reads_the_verdict_the_summary_writes(self) -> None:
        from tests.studies.evidence import property_failures as pf

        for rates in ({}, {(study.RELATIVE_FAMILY, "I2-T__placebo"): 0.11}):
            rows = _rows(rates)
            summary = properties.summarize_properties(rows)
            ctx = pf.Context(study.STUDY, rows, summary)
            for rule in properties.PROPERTY_RULES:
                for _, row in summary.loc[summary["property"].isin(rule.families)].iterrows():
                    legs = [
                        clause.read(ctx, row).passed
                        for clause in rule.clauses
                        if clause.target == "passed" and clause.applies(ctx, row)
                    ]
                    if legs:
                        assert all(legs) == bool(row["passed"]), (row["property"], row["cell"])

    def test_the_relative_leg_enumerates_both_cells_lost_trials(self) -> None:
        from tests.studies.evidence import property_failures as pf

        rows = _rows({})
        summary = properties.summarize_properties(rows)
        lost = {
            (study.RELATIVE_FAMILY, "I2-T__placebo"): 2,
            (study.RELATIVE_FAMILY, f"I2-T__placebo{properties.PARTNER}"): 3,
        }
        ctx = pf.Context(study.STUDY, rows, summary, lost=lost)
        rule = pf.rule_for(study.RELATIVE_FAMILY, properties)
        assert rule is not None
        pick = (summary["property"] == study.RELATIVE_FAMILY) & (summary["cell"] == "I2-T__placebo")
        row = summary.loc[pick].iloc[0]
        assert len(rule.clauses[0].candidates(ctx, row)) == 3 * 4

    def test_every_gated_family_resolves_to_a_rule(self) -> None:
        from tests.studies.evidence import property_failures as pf

        for family in study.PROPERTY_CELLS:
            assert pf.rule_for(family, properties) is not None
