"""The ``ep-cate`` declaration: laws, seeds, cells, verdicts and the computed design numbers.

The laws are checked against hte3's code (``generate_data_CATE.R`` and
``generate_data_CATE_HighDim.R`` at ``eac64673``) at fixed points; the declared ``q_bounds``
against a large draw; the replicate counts against the plan's sizing rule at the committed
pilot (:func:`tests.studies.ep_cate.design_numbers`); and the verdict code against synthetic
rows, including a control that must fail and does when the rows say it should.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tests.studies import ep_cate as study
from tests.studies import ep_cate_properties as properties
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.seeds import stream_seed

LAW = study.LAW_BY_LABEL


class TestTheLawsAreHte3s:
    POINT = np.array([[0.3, -0.5, 0.8]])

    def test_the_d3_propensities(self) -> None:
        total = self.POINT.sum()
        assert study.propensity(LAW["d3_simple_limited"], self.POINT)[0] == pytest.approx(
            1 / (1 + math.exp(-total))
        )
        assert study.propensity(LAW["d3_simple_moderate"], self.POINT)[0] == pytest.approx(
            1 / (1 + math.exp(-total / 3))
        )

    def test_the_d3_cates(self) -> None:
        w = self.POINT
        assert study.cate(LAW["d3_simple_moderate"], w)[0] == pytest.approx(1 + 0.3)
        assert study.cate(LAW["d3_simple_limited"], w)[0] == pytest.approx(1 + w.sum())
        complex_ = 1 + w.sum() + np.sin(5 * w).sum()
        for label in ("d3_complex_moderate", "d3_complex_limited"):
            assert study.cate(LAW[label], w)[0] == pytest.approx(complex_)

    def test_the_d3_baseline(self) -> None:
        w = self.POINT
        expected = 0.5 * w.sum() + np.sin(5 * w).sum() + (1 / (w + 1.2)).sum()
        assert study.baseline(LAW["d3_complex_limited"], w)[0] == pytest.approx(expected)

    def test_the_d20_law(self) -> None:
        w = np.linspace(-0.9, 0.9, 20)[None, :]
        law = LAW["d20_complex"]
        w1, w5, w9, w10, w11, w15, w19 = (w[0, j] for j in (0, 4, 8, 9, 10, 14, 18))
        assert study.propensity(law, w)[0] == pytest.approx(
            1 / (1 + math.exp(-(w1 + w5 + w9 + w11 + w19) / 1.3))
        )
        assert study.cate(law, w)[0] == pytest.approx(
            1
            + (math.sin(4 * w1) + math.sin(4 * w5) + math.cos(4 * w9) + 1.5 * (w15**2 - w10**2)) / 5
        )
        assert study.cate(LAW["d20_simple"], w)[0] == pytest.approx(
            1 + (w1 + w5 + w9 + w15 + w10) / 5
        )
        base = (
            math.cos(4 * w1)
            + math.cos(4 * w5)
            + math.sin(4 * w9)
            + 1 / (1.5 + w15)
            + 1 / (1.5 + w10)
        ) / 5 + (math.sin(5 * w1) + math.sin(5 * w5) + math.sin(5 * w9)) / 2
        assert study.baseline(law, w)[0] == pytest.approx(base)

    def test_the_d20_covariates_are_the_truncated_equicorrelated_normal_over_two(self) -> None:
        w = study.covariates(LAW["d20_simple"], 20_000, np.random.default_rng(0))
        assert np.max(np.abs(w)) <= 1.0
        correlation = np.corrcoef(w[:, 0], w[:, 1])[0, 1]
        # Truncating all 20 coordinates to [-2, 2] mostly removes draws with a large common
        # factor, so the 0.4 correlation shrinks (to about 0.19 at this draw); a draw without
        # the truncation would keep 0.4 and one without the common factor would read 0.
        assert 0.1 < correlation < 0.3
        assert abs(float(np.mean(w))) < 0.01

    def test_the_noise_truncation_moves_the_law_by_its_tail_mass(self) -> None:
        assert 2 * stats.norm.sf(study.NOISE_TRUNCATION) == pytest.approx(6.3e-5, rel=0.01)
        assert 2 * stats.norm.sf(study.RESCUE_NOISE_TRUNCATION / study.RESCUE_NOISE_SD) == (
            pytest.approx(6.3e-5, rel=0.01)
        )

    def test_the_rescue_laws_carry_app_b2_noise_and_otherwise_the_code_law(self) -> None:
        for overlap in ("moderate", "limited"):
            rescue, code = LAW[f"d3_complex_{overlap}_noise2"], LAW[f"d3_complex_{overlap}"]
            assert (rescue.noise_sd, rescue.noise_truncation) == (2.0, 4.0 * 2.0)
            assert (code.noise_sd, code.noise_truncation) == (1.0, 4.0)
            w = self.POINT
            for part in (study.propensity, study.baseline, study.cate):
                assert part(rescue, w)[0] == part(code, w)[0]
            # A nonzero witness: the drawn noise's SD is the declared one.
            frame = study.sample(rescue, 40_000, np.random.default_rng(2))
            x = frame.loc[:, list(rescue.columns)].to_numpy()
            noise = frame["Y"] - study.baseline(rescue, x) - frame["A"] * study.cate(rescue, x)
            assert float(np.std(noise)) == pytest.approx(2.0, rel=0.02)

    @pytest.mark.parametrize("label", [law.label for law in study.LAWS])
    def test_q_bounds_cover_a_large_draw(self, label: str) -> None:
        law = LAW[label]
        frame = study.sample(law, 50_000, np.random.default_rng(1))
        low, high = study.q_bounds(law)
        assert low < frame["Y"].min() and frame["Y"].max() < high


class TestTheDeclaration:
    def test_the_schedule(self) -> None:
        assert [study.schedule(n) for n in (1_000, 2_000, 4_000)] == [3, 4, 5]
        assert study.schedule(4_000) <= study.SIEVE_MAX

    def test_every_gated_law_has_its_positive_rows_and_the_rescue_its_control(self) -> None:
        for law in study.LAWS:
            group = study.GROUP_BY_LABEL[study.group_label(law.label, law.sizes[-1])]
            roles = study.oracle_cells(group)
            if law.ensemble:
                assert roles[f"{group.label}__ep"] == "positive"
            else:
                assert f"{group.label}__ep" not in roles
            if law.rescue:
                assert roles[f"{group.label}__ep_misspecified"] == "positive"
                assert roles[f"{group.label}__t_misspecified"] == "control"
        assert {law.label for law in study.LAWS if law.rescue} == {
            "d3_complex_moderate_noise2",
            "d3_complex_limited_noise2",
        }

    def test_the_code_law_rescue_is_a_diagnostic_at_the_largest_n_only(self) -> None:
        for label in ("d3_complex_moderate", "d3_complex_limited"):
            law = LAW[label]
            assert not law.rescue and law.misspecified == "diagnostic"
            large = study.oracle_cells(study.GROUP_BY_LABEL[study.group_label(label, 4_000)])
            assert large[f"{label}_4k__ep_misspecified"] == study.DIAGNOSTIC
            assert large[f"{label}_4k__t_misspecified"] == study.DIAGNOSTIC
            small = study.oracle_cells(study.GROUP_BY_LABEL[study.group_label(label, 1_000)])
            assert f"{label}_1k__ep_misspecified" not in small

    def test_the_ungated_middle_n_runs_the_descriptive_count(self) -> None:
        for law in study.LAWS:
            for n in law.sizes:
                group = study.GROUP_BY_LABEL[study.group_label(law.label, n)]
                expected = (
                    study.REPLICATES[law.label]
                    if n in study.gated_n(law)
                    else study.DESCRIPTIVE_REPLICATES
                )
                assert group.replicates == expected

    def test_the_streams_are_disjoint(self) -> None:
        seeds = {
            stream_seed(study.STUDY, stream, group.label, r)
            for stream in ("property_sample", "ise_ratio_pilot", "rescue_noise_pilot")
            for group in study.GROUPS
            for r in range(3)
        }
        seeds |= {stream_seed(study.STUDY, "ise_points", law.label) for law in study.LAWS}
        assert len(seeds) == 3 * len(study.GROUPS) * 3 + len(study.LAWS)
        assert study.SEED != study.RESAMPLING_SEED

    def test_the_fixed_cell_reads_the_projection_law(self) -> None:
        assert study.FIXED_N == 4_000
        assert study.FIXED_REPLICATES == 2_400
        assert set(study.fixed_truth()) == {"(intercept)", "W1", "W3"}

    def test_the_comparator_is_declared_pending(self) -> None:
        assert study.COMPARATOR.status == "pending"
        assert study.COMPARATOR.kind == "different estimator"
        assert "targeting_style = 'dr'" in study.COMPARATOR.learner
        runner = study.STUDY.artifacts / study.COMPARATOR.runner
        assert runner.is_file()
        assert 'targeting_style = "dr"' in runner.read_text()


class TestTheDesignNumbers:
    @pytest.fixture(scope="class")
    def numbers(self) -> dict:
        return study.design_numbers()

    @pytest.mark.parametrize("label", [law.label for law in study.LAWS])
    def test_the_declared_r_is_the_sizing_rules(self, numbers: dict, label: str) -> None:
        assert study.REPLICATES[label] == numbers["laws"][label]["replicates"]

    def test_every_powered_law_reaches_the_target(self, numbers: dict) -> None:
        for label, row in numbers["laws"].items():
            if row["powered"]:
                assert min(row["gates"].values()) >= study.TARGET_PASS, label

    def test_every_law_is_powered(self, numbers: dict) -> None:
        assert all(row["powered"] for row in numbers["laws"].values())

    def test_the_code_law_rescue_would_not_pass_and_the_app_b2_one_does(self) -> None:
        """The first pilot's finding, which moved the rescue gate (plan amendment rev 2.4).

        Under hte3's noise SD 1 the misspecified EP learner sits near ``log 4.3`` above the
        oracle at 4k, so a gate at ``log 3`` fails at any R; under App B.2's SD 2 it sits near
        ``log 2``.  Both readings are pilot means, on disjoint streams.
        """
        pilot = study.pilot_rows()
        for overlap in ("moderate", "limited"):
            code = pilot[(f"d3_complex_{overlap}_4k", "ep_misspecified")]
            assert (
                study.gate_pass(code["mean"], code["sd"], 600, study.RESCUE_MARGIN, below=True)
                < 1e-6
            )
            noisy = pilot[(f"d3_complex_{overlap}_noise2_4k", "ep_misspecified")]
            assert noisy["mean"] < study.RESCUE_MARGIN - 0.3

    def test_the_fixed_size_rows_pass_with_high_probability(self, numbers: dict) -> None:
        for row in numbers["fixed"].values():
            assert row["pass"] > 0.99

    def test_the_gate_probability_is_monotone_in_the_mean(self) -> None:
        low = study.gate_pass(0.1, 0.3, 300, study.THM5_MARGIN, below=True)
        high = study.gate_pass(0.40, 0.3, 300, study.THM5_MARGIN, below=True)
        assert low > 0.99 > 0.5 > high


def synthetic_rows(ep_mean: float, control_mean: float) -> pd.DataFrame:
    """Rows of every declared cell at constant spread, with the gated means given."""
    rng = np.random.default_rng(3)
    rows = []
    for group in study.GROUPS:
        for family, cells in study.group_cells(group).items():
            for cell in cells:
                role = (
                    study.oracle_cells(group).get(cell, property_verdicts.DIAGNOSTIC_ROLE)
                    if family == study.ORACLE
                    else ("positive" if family == study.BIAS else property_verdicts.DIAGNOSTIC_ROLE)
                )
                mean = 0.0
                if cell.endswith("__t_misspecified"):
                    mean = control_mean
                elif cell.endswith("__ep") or cell.endswith("__ep_misspecified"):
                    mean = ep_mean
                # The bias endpoint needs about 200 replicates to resolve +-0.25.
                count = 400 if family == study.BIAS else 40
                for r in range(count):
                    rows.append(
                        {
                            "property": family,
                            "cell": cell,
                            "role": role,
                            "replicate": r,
                            "n": group.n,
                            "requested_replicates": count,
                            "failed_replicates": 0,
                            "truth": 0.0,
                            "estimate": mean + 0.1 * rng.standard_normal(),
                            "std_error": 1.0,
                            "covered": 1,
                            "rejected": 0,
                        }
                    )
    return pd.DataFrame(rows)


class TestTheVerdicts:
    def verdicts(self, ep_mean: float, control_mean: float) -> pd.DataFrame:
        return properties.summarize_properties(synthetic_rows(ep_mean, control_mean))

    def test_good_rows_pass(self) -> None:
        summary = self.verdicts(0.05, 3.0)
        assert summary["passed"].all()
        growth = summary.loc[summary["cell"].str.endswith("_growth")]
        assert len(growth) == sum(law.ensemble + law.rescue for law in study.LAWS)

    def test_an_ep_above_the_margin_fails(self) -> None:
        summary = self.verdicts(1.2, 3.0)
        failed = set(summary.loc[~summary["passed"], "cell"])
        assert "d3_simple_moderate_4k__ep" in failed
        assert "d3_complex_limited_noise2_4k__ep_misspecified" in failed
        # The code-law rows are diagnostic: they state no verdict at any value.
        assert "d3_complex_limited_4k__ep_misspecified" not in failed

    def test_a_control_that_does_not_detect_fails(self) -> None:
        summary = self.verdicts(0.05, 0.5)
        failed = set(summary.loc[~summary["passed"], "cell"])
        assert failed == {
            "d3_complex_moderate_noise2_4k__t_misspecified",
            "d3_complex_limited_noise2_4k__t_misspecified",
        }

    def test_growth_reads_the_change_from_the_smallest_to_the_largest_n(self) -> None:
        rows = synthetic_rows(0.05, 3.0)
        at_large = rows["cell"] == "d3_simple_moderate_4k__ep"
        rows.loc[at_large, "estimate"] += 0.3
        summary = properties.summarize_properties(rows)
        growth = summary.set_index("cell").loc["d3_simple_moderate__ep_growth"]
        assert not bool(growth["passed"])
        assert growth["ratio_mean"] == pytest.approx(0.3, abs=0.05)
