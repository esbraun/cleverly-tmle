"""The ``ep-crr`` declaration: the law, the oracle, the cells, the verdicts and the design numbers.

The law is checked against hte3's ``generate_data_RR.R`` at ``eac64673`` at a fixed point; the
oracle's Newton solver against an exact log-ratio fit and a non-convex witness; the replicate
counts against the sizing rule at the committed pilot; and the verdict code against synthetic
rows, including the exclusion of non-positive-definite oracle replicates.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from tests.studies import ep_cate
from tests.studies import ep_crr as study
from tests.studies import ep_crr_properties as properties
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.seeds import stream_seed

LAW = study.LAW_BY_LABEL
POINT = np.array([[0.3, -0.5, 0.8]])


class TestTheLawIsHte3s:
    def test_the_baseline_and_the_log_crrs(self) -> None:
        w = POINT
        total, waves = w.sum(), np.sin(4 * w).sum()
        assert study.baseline(w)[0] == pytest.approx(float(expit(-1 + 0.3 * (total + waves))))
        assert study.log_crr(LAW["d3_simple_limited"], w)[0] == pytest.approx(0.1 * (-1 + total))
        assert study.log_crr(LAW["d3_complex_moderate"], w)[0] == pytest.approx(
            0.1 * (-1 + total + waves)
        )

    def test_the_propensities(self) -> None:
        total = POINT.sum()
        assert study.propensity(LAW["d3_simple_limited"], POINT)[0] == pytest.approx(
            float(expit(total))
        )
        assert study.propensity(LAW["d3_simple_moderate"], POINT)[0] == pytest.approx(
            float(expit(total / 3))
        )

    @pytest.mark.parametrize("label", list(LAW))
    def test_nothing_is_clipped(self, label: str) -> None:
        # hte3 clips mu(1, w) at 1; the declared check says no point of the domain reaches it.
        assert study.max_treated_mean(LAW[label]) < 0.8

    def test_a_draw_reads_the_declared_means(self) -> None:
        law = LAW["d3_complex_limited"]
        frame = study.sample(law, 200_000, np.random.default_rng(4))
        treated = frame["A"] == 1
        w = frame.loc[treated, list(law.columns)].to_numpy()
        assert frame.loc[treated, "Y"].mean() == pytest.approx(
            float(np.mean(study.treated_mean(law, w))), abs=0.005
        )


class TestTheOracle:
    def test_the_plug_in_fit_is_exact_when_the_basis_holds_the_log_crr(self) -> None:
        law = LAW["d3_simple_moderate"]
        w = study.covariates(5_000, np.random.default_rng(5))
        design = np.column_stack([np.ones(len(w)), w])
        beta, positive = study.oracle_fit(
            design, study.treated_mean(law, w), study.baseline(w), np.zeros(4)
        )
        assert positive
        assert beta == pytest.approx([-0.1, 0.1, 0.1, 0.1], abs=1e-8)

    def test_the_solution_is_stationary_from_the_plug_in_start(self) -> None:
        law = LAW["d3_complex_limited"]
        frame = study.sample(law, 4_000, np.random.default_rng(6))
        w = frame.loc[:, list(law.columns)].to_numpy()
        design = np.column_stack([np.ones(len(w)), w, np.sin(4 * w)])
        start, _ = study.oracle_fit(
            design, study.treated_mean(law, w), study.baseline(w), np.zeros(7)
        )
        d_t, d_r = study.oracle_pseudo(law, frame)
        beta, positive = study.oracle_fit(design, d_t, d_r, start)
        p = expit(design @ beta)
        gradient = design.T @ ((d_t + d_r) * p - d_t) / len(p)
        assert positive
        assert np.max(np.abs(gradient)) < 1e-9

    def test_a_negative_weight_hessian_is_flagged(self) -> None:
        # A nonzero witness of the check: pseudo-outcomes whose sum is negative on most rows
        # give a negative-definite Hessian, and the flag must say so.
        rng = np.random.default_rng(7)
        design = np.column_stack([np.ones(200), rng.uniform(-1, 1, 200)])
        d_t = -np.abs(rng.normal(size=200))
        d_r = -np.abs(rng.normal(size=200))
        _, positive = study.oracle_fit(design, d_t, d_r, np.zeros(2))
        assert not positive


class TestTheDeclaration:
    def test_the_roles(self) -> None:
        for law in study.LAWS:
            large = study.oracle_cells(study.GROUP_BY_LABEL[study.group_label(law.label, 4_000)])
            assert large[f"{law.label}_4k__ep"] == "positive"
            assert (f"{law.label}_4k__ep_misspecified" in large) == law.rescue
        rescue = study.oracle_cells(study.GROUP_BY_LABEL["d3_complex_limited_4k"])
        for learner in ("misspecified", "offset"):
            assert rescue[f"d3_complex_limited_4k__ep_{learner}"] == "positive"
            assert rescue[f"d3_complex_limited_4k__t_{learner}"] == "control"
        # One Hessian check per replicate: every learner's oracle is the same fit.
        assert study.hessian_cells(study.GROUP_BY_LABEL["d3_complex_limited_4k"]) == (
            "d3_complex_limited_4k__oracle_pd",
        )
        assert [law.label for law in study.LAWS if law.rescue] == ["d3_complex_limited"]
        assert study.oracle_cells(study.GROUP_BY_LABEL[study.CV_GROUP]) == {
            f"{study.CV_GROUP}__ep": ep_cate.DIAGNOSTIC
        }

    def test_the_margins_are_the_plans(self) -> None:
        assert math.isclose(study.RESCUE_MARGIN, math.log(1.5))
        assert math.isclose(study.THM5_MARGIN, math.log(1.5))
        assert math.isclose(study.GROWTH_MARGIN, math.log(1.25))

    def test_the_streams_and_seeds_are_disjoint(self) -> None:
        seeds = {
            stream_seed(study.STUDY, stream, group.label, r)
            for stream in (
                "property_sample",
                "crr_ise_ratio_pilot",
                "crr_growth_pilot",
                "crr_hessian_pilot",
                "b1_offset_pilot_scale2",
            )
            for group in study.GROUPS
            for r in range(3)
        }
        seeds |= {stream_seed(study.STUDY, "ise_points", law.label) for law in study.LAWS}
        assert len(seeds) == 5 * len(study.GROUPS) * 3 + len(study.LAWS)
        others = {ep_cate.SEED, ep_cate.RESAMPLING_SEED}
        assert {study.SEED, study.RESAMPLING_SEED}.isdisjoint(others)
        assert study.SEED != study.RESAMPLING_SEED

    def test_the_comparator_is_declared_pending(self) -> None:
        assert study.COMPARATOR.status == "pending"
        assert "Lrnr_crr_EP" in study.COMPARATOR.learner
        runner = study.STUDY.artifacts / study.COMPARATOR.runner
        assert runner.is_file()
        assert "Lrnr_crr_EP$new(" in runner.read_text()


class TestTheDesignNumbers:
    @pytest.fixture(scope="class")
    def numbers(self) -> dict:
        return study.design_numbers()

    @pytest.mark.parametrize("label", [law.label for law in study.LAWS])
    def test_the_declared_r_is_the_sizing_rules(self, numbers: dict, label: str) -> None:
        assert study.REPLICATES[label] == numbers["laws"][label]["replicates"]

    def test_every_law_is_powered(self, numbers: dict) -> None:
        # The N2 rule reads each gate's assurance (design review B2 of PR-b).
        for label, row in numbers["laws"].items():
            assert row["powered"], label
            assert ep_cate.min_gate(row["assurance"]) >= study.TARGET_PASS, label
            assert min(row["gates"].values()) >= study.TARGET_PASS, label

    def test_the_family_wise_assurance(self, numbers: dict) -> None:
        family = numbers["family"]
        assert family["pass"] > 0.9
        assert family["no_suspect"] > 0.95

    def test_the_offset_rescue_sees_the_propensity_weight(self, numbers: dict) -> None:
        # Design review B1: the unit-weight mutation fails the offset rescue gate.
        assert numbers["mutation_detected"] > 0.999

    def test_the_offset_learner_is_the_truth_off_by_the_declared_error(self) -> None:
        law = LAW["d3_complex_limited"]
        w = study.covariates(50, np.random.default_rng(9))
        learner = study.OffsetOutcome().fit(np.column_stack([np.ones(50), w]), np.zeros(50))
        for arm, expected in (
            (
                1.0,
                expit(
                    np.log(study.treated_mean(law, w) / (1 - study.treated_mean(law, w)))
                    + 2.0 * (w.sum(axis=1) ** 2 - 1.0)
                ),
            ),
            (0.0, study.baseline(w)),
        ):
            design = np.column_stack([np.full(50, arm), w])
            assert learner.predict_proba(design)[:, 1] == pytest.approx(expected)
        # The error is centred under the covariate law, and most of it lies outside any
        # additive model: its cross terms 4 sum_{j<k} W_j W_k survive a regression on
        # polynomials of each W_k alone.
        draws = study.covariates(200_000, np.random.default_rng(10))
        error = study.offset_error(draws)
        assert abs(np.mean(error)) < 0.03
        additive = np.column_stack(
            [np.ones(len(draws)), *(draws[:, [k]] ** p for k in range(3) for p in (1, 2, 3, 4))]
        )
        residual = error - additive @ np.linalg.lstsq(additive, error, rcond=None)[0]
        assert np.var(residual) > 0.7 * np.var(error)

    def test_the_oracle_hessian_stays_within_the_failure_bound(self, numbers: dict) -> None:
        # The oracle-only pilot ran 3,000 checks per law and n and saw no non-PD Hessian.  At
        # the Jeffreys rate and at its one-sided 95% upper limit, every gated group stays
        # within floor(0.01 R) together (design review S1).
        hessian = numbers["hessian"]
        assert hessian["pilot"]["checks"] == 3_000 * 2 * len(study.LAWS)
        assert hessian["pilot"]["failures"] == 0
        for label, probability in hessian["pass"].items():
            assert probability > 0.999, label
        assert hessian["family_pass"] > 0.999
        assert hessian["family_pass_upper"] > 0.999

    def test_the_nuisance_product_falls_faster_than_root_n(self, numbers: dict) -> None:
        for label, row in numbers["nuisance"].items():
            assert row["faster_than_root_n"], label


def synthetic_rows(ep_mean: float, control_mean: float, *, non_pd: int = 0) -> pd.DataFrame:
    """Rows of every declared cell, with the gated means given.

    ``non_pd`` replicates of ``d3_complex_limited_4k`` read a non-PD oracle Hessian, and their
    EP rescue ratios are set far above the margin, so only their exclusion keeps the row green.
    """
    rng = np.random.default_rng(8)
    rows = []
    for group in study.GROUPS:
        roles = study.oracle_cells(group)
        for family, cells in study.group_cells(group).items():
            for cell in cells:
                role = roles.get(cell, property_verdicts.DIAGNOSTIC_ROLE)
                for r in range(group.replicates):
                    bad = group.label == "d3_complex_limited_4k" and r < non_pd
                    if family == study.HESSIAN:
                        value = 0.0 if bad else 1.0
                    elif cell.endswith(("__t_misspecified", "__t_offset")):
                        value = control_mean + 0.1 * rng.standard_normal()
                    elif cell.endswith(("__ep", "__ep_misspecified", "__ep_offset")):
                        value = 10.0 if bad else ep_mean + 0.1 * rng.standard_normal()
                    else:
                        value = 0.1 * rng.standard_normal()
                    rows.append(
                        {
                            "property": family,
                            "cell": cell,
                            "role": role,
                            "replicate": r,
                            "n": group.n,
                            "requested_replicates": group.replicates,
                            "failed_replicates": 0,
                            "truth": 0.0,
                            "estimate": value,
                            "std_error": 1.0,
                            "covered": 1,
                            "rejected": 0,
                        }
                    )
    return pd.DataFrame(rows)


class TestTheVerdicts:
    def failed(self, rows: pd.DataFrame) -> set[str]:
        summary = properties.summarize_properties(rows)
        return set(summary.loc[~summary["passed"].astype(bool), "cell"])

    def test_good_rows_pass(self) -> None:
        summary = properties.summarize_properties(synthetic_rows(0.05, 2.0))
        assert summary["passed"].astype(bool).all()
        growth = summary.loc[summary["cell"].str.endswith("_growth")]
        assert len(growth) == len(study.LAWS) + len(study.RESCUE_LEARNERS)

    def test_an_ep_above_the_margin_fails(self) -> None:
        failed = self.failed(synthetic_rows(0.6, 2.0))
        assert "d3_simple_moderate_4k__ep" in failed
        assert "d3_complex_limited_4k__ep_misspecified" in failed
        assert "d3_complex_limited_4k__ep_offset" in failed

    def test_the_offset_growth_is_reported_not_gated(self) -> None:
        # Plan amendment rev 2.5: both rescue ratios grow by 0.35 from 1k to 4k, past the
        # growth margin; only the intercept learner's growth row is a gate.
        rows = synthetic_rows(0.05, 2.0)
        small = rows["cell"].isin(
            ["d3_complex_limited_1k__ep_misspecified", "d3_complex_limited_1k__ep_offset"]
        )
        rows.loc[small, "estimate"] -= 0.35
        summary = properties.summarize_properties(rows)
        failed = set(summary.loc[~summary["passed"].astype(bool), "cell"])
        assert failed == {"d3_complex_limited__ep_misspecified_growth"}
        offset = summary.set_index("cell").loc["d3_complex_limited__ep_offset_growth"]
        assert offset["role"] == property_verdicts.DIAGNOSTIC_ROLE
        assert offset["ratio_upper"] > study.GROWTH_MARGIN
        assert "rescue_offset_growth" not in study.assured(
            study.LAW_BY_LABEL["d3_complex_limited"], study.pilot_rows(), 300
        )

    def test_a_control_that_does_not_detect_fails(self) -> None:
        assert self.failed(synthetic_rows(0.05, 0.2)) == {
            "d3_complex_limited_4k__t_misspecified",
            "d3_complex_limited_4k__t_offset",
        }

    def test_the_ratio_rows_carry_no_bias_or_coverage_reading(self) -> None:
        # Design review S3: those shared columns mean nothing on a log ratio with unit SE.
        summary = properties.summarize_properties(synthetic_rows(0.05, 2.0))
        oracle = summary.loc[summary["property"] == study.ORACLE]
        for column in ("standardized_bias", "bias", "coverage", "se_ratio"):
            assert oracle[column].isna().all(), column
        assert oracle["ratio_mean"].notna().all()

    def test_non_pd_replicates_are_excluded_within_the_bound(self) -> None:
        allowed = math.floor(study.FAILURE_BOUND * study.REPLICATES["d3_complex_limited"])
        assert not self.failed(synthetic_rows(0.05, 2.0, non_pd=allowed))

    def test_more_non_pd_replicates_than_the_bound_turn_the_group_red(self) -> None:
        allowed = math.floor(study.FAILURE_BOUND * study.REPLICATES["d3_complex_limited"])
        failed = self.failed(synthetic_rows(0.05, 2.0, non_pd=allowed + 1))
        assert "d3_complex_limited_4k__ep" in failed
        assert "d3_complex_limited__ep_growth" in failed
