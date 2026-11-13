"""The design of ``option-stopping-ltmle`` (Fix-8 plan sections 5.0 and 5.2), recomputed.

Every declared number of :mod:`tests.studies.canonical_option_stopping_ltmle` and
:mod:`tests.studies.option_stopping_ltmle_properties` is computed here, from the exact laws and
from the committed pilot (:mod:`tests.studies.option_stopping_pilot`, 2,000 draws per cell):
the truths, the pre-read of every gated cell, the replications, the pass probabilities of
every control and of W and N, the noise arm's scale, and that the threshold patch reaches a
hazard walk.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

import cleverly.fluctuation._score as score_module
from tests.studies import _option_stopping_cells as grid
from tests.studies import _option_stopping_laws as laws
from tests.studies import canonical_option_stopping_ltmle as study
from tests.studies import mtp_study_tools as tools
from tests.studies import option_stopping_ltmle_properties as properties
from tests.studies import option_stopping_pilot as pilot
from tests.studies import repeated_sizing as sizing

MARGINS = study.STUDY.margins
PRE_READ_COVERAGE = 0.937
PRE_READ_WITNESS = 0.7
N_CELL = "mechanism_correct_sv_one_step_K8"
#: 12 calibration cells, the gated W cell and the N pair.
GATING_COUNT = 15


@pytest.fixture(scope="module")
def summary() -> pd.DataFrame:
    return pilot.read_summary()


def reading(summary: pd.DataFrame, law: str, cell: str, parameter: str) -> pd.Series:
    rows = summary.loc[
        (summary["study"] == "ltmle")
        & (summary["law"] == law)
        & (summary["cell"] == cell)
        & (summary["parameter"] == parameter)
    ]
    assert len(rows) == 1, (law, cell, parameter)
    return rows.iloc[0]


def label_reading(summary: pd.DataFrame, label: str) -> pd.Series:
    """The pilot row of a calibration label such as ``l4_sv_one_step``."""
    stream, fit = label.split("_", 1)
    law = stream.upper()
    return reading(summary, law, fit, study.CONTRASTS[law])


class TestTheLaws:
    def test_the_truths_do_not_depend_on_the_draw(self) -> None:
        truths = study.truths()
        assert laws.draw_l3(30, 11)[1][laws.L3_CONTRAST] == truths["L3"][laws.L3_CONTRAST]
        assert laws.draw_l4(30, 11)[1][laws.L4_RISK] == truths["L4"][laws.L4_RISK]

    def test_the_l4_truths_are_point_survival_curve_s(self) -> None:
        from tests.studies import canonical_point_survival_curve as curve

        truth = curve.TRUTH[curve.SCENARIO]
        assert study.truths()["L4"][laws.L4_CONTRAST] == truth[laws.L4_CONTRAST]
        assert study.truths()["L4"][laws.L4_RISK] == truth[laws.L4_RISK]


class TestTheDeclaration:
    def test_the_constants(self) -> None:
        assert study.PRIMARY_N == laws.N == 2_000
        assert (study.WITNESS_SHARE, study.NECESSITY_CEILING, study.NECESSITY_FLOOR) == (
            0.5,
            0.92,
            0.90,
        )
        assert laws.L4_STEP_SIZE == 1e-3
        assert study.STUDY.publication_policy == "reporting"
        assert study.STUDY.reference is None

    def test_the_seeds_are_the_study_s_own(self) -> None:
        from tests.studies import canonical_option_stopping_point as sibling
        from tests.studies.evidence.registry import registered

        seeds = {record.seed for record in registered()} | {sibling.SEED, pilot.PILOT_SEED}
        assert study.SEED not in seeds
        assert study.RESAMPLING_SEED not in {r.resampling_seed for r in registered()}

    def test_the_record_declares_the_cells_the_module_builds(self) -> None:
        assert study.STUDY.property_cells == properties.PROPERTY_CELLS
        built = {(c.property, c.cell) for c in properties.declared_cells()}
        declared = {(f, c) for f, cells in study.PROPERTY_CELLS.items() for c in cells}
        assert built == declared

    def test_four_gated_calibration_positives(self) -> None:
        assert study.CALIBRATION_LABELS == (
            "l3_sv_newton",
            "l3_ss_newton",
            "l4_sv_one_step",
            "l4_ss_one_step",
        )
        assert len(study.PROPERTY_CELLS["interval_calibration"]) == 12

    def test_the_gated_witness_and_necessity_cells(self) -> None:
        roles = {(c.family, c.name): c.role for c in study.CELLS}
        gated = {
            name
            for (family, name), role in roles.items()
            if family == "residual_witness" and role == "positive"
        }
        assert gated == {"l4_sv_one_step__witness"}
        assert roles[("residual_necessity", "l4_mc_risk__widened")] == "positive"
        assert roles[("residual_necessity", "l4_mc_risk__plain")] == "control"
        necessity = [c for c in study.CELLS if c.family == "residual_necessity"]
        assert {c.fit for c in necessity} == {f"mc_sv_one_step_k{laws.NECESSITY_SCALE:g}"}
        assert {c.estimand for c in necessity} == {laws.L4_RISK}


class TestThePreRead:
    def test_the_pilot_ran_at_least_2000_draws_per_cell(self, summary: pd.DataFrame) -> None:
        assert (summary.loc[summary["study"] == "ltmle", "draws"] >= pilot.DRAWS).all()

    @pytest.mark.parametrize("label", study.CALIBRATION_LABELS)
    def test_every_gated_calibration_cell_reads_coverage_above_the_pre_read_floor(
        self, summary: pd.DataFrame, label: str
    ) -> None:
        row = label_reading(summary, label)
        assert row["coverage"] >= PRE_READ_COVERAGE
        assert (
            sizing.calibration_pass(MARGINS, study.REPLICATES, np.array([row["se_ratio"]]))[0] > 0.5
        )

    def test_the_gated_witness_reads_a_share_of_at_least_0_7(self, summary: pd.DataFrame) -> None:
        assert label_reading(summary, "l4_sv_one_step")["residual_share"] >= PRE_READ_WITNESS

    def test_the_rule_cells_meet_their_rule(self, summary: pd.DataFrame) -> None:
        rows = summary.loc[summary["study"] == "ltmle"]
        assert (rows["rule_met_share"] >= 0.99).all()

    @pytest.mark.parametrize("fit", ["sv_newton", "ss_newton"])
    def test_the_l3_newton_walk_lands_at_the_relative_fit(
        self, summary: pd.DataFrame, fit: str
    ) -> None:
        # The rule never binds on L3's per-node Newton walk, so its cells carry the relative
        # fit's evidence and license only that walk (design review S2).
        ruled = reading(summary, "L3", fit, laws.L3_CONTRAST)
        relative = reading(summary, "L3", "rel_newton", laws.L3_CONTRAST)
        assert ruled["coverage"] == relative["coverage"]
        assert ruled["se_ratio"] == pytest.approx(relative["se_ratio"], rel=1e-9)
        assert ruled["residual_share"] == 0.0
        assert ruled["mean_abs_residual"] < 1e-9


class TestTheSizes:
    def test_the_product_floor(self) -> None:
        def product(replicates: int) -> float:
            cell = float(sizing.calibration_pass(MARGINS, replicates, np.array([1.0]))[0])
            return cell**4

        floor = next(r for r in range(400, 20_001, 400) if product(r) >= 0.95)
        assert floor == 2_800
        assert product(floor) == pytest.approx(0.969, abs=1e-3)

    def test_the_pooled_predictive_rule_sets_the_replications(self, summary: pd.DataFrame) -> None:
        ratios = {
            f"{label}__widened": float(label_reading(summary, label)["se_ratio"])
            for label in study.CALIBRATION_LABELS
        }

        def correlations(labels: Any) -> np.ndarray:
            # Cells on one law read the same draws (pilot correlation of their estimates at
            # least 0.999), so they are taken as perfectly correlated; the laws share none.
            streams = [label.split("_", 1)[0] for label in labels]
            return np.array([[float(a == b) for b in streams] for a in streams])

        pooled = sizing.calibration_replicates(MARGINS, ratios, pilot.DRAWS, correlations)
        assert max(2_800, pooled) == study.REPLICATES

    def test_the_derived_controls_fail_with_certainty(self) -> None:
        for ratio in (properties.SHRUNKEN_SE_FACTOR, 1.0 / np.sqrt(2.0)):
            assert sizing.calibration_control_probability(
                MARGINS, study.REPLICATES, ratio
            ) == pytest.approx(1.0, abs=1e-6)

    def test_the_witness_passes_at_the_pilot_share(self, summary: pd.DataFrame) -> None:
        share = float(label_reading(summary, "l4_sv_one_step")["residual_share"])
        assert grid.floor_pass_probability(
            MARGINS, study.REPLICATES, share, study.WITNESS_SHARE
        ) == pytest.approx(1.0, abs=1e-6)

    def test_the_necessity_pair_passes_at_the_pilot_readings(self, summary: pd.DataFrame) -> None:
        row = reading(summary, "L4", N_CELL, laws.L4_RISK)
        assert grid.ceiling_pass_probability(
            MARGINS, study.REPLICATES, float(row["plain_coverage"]), study.NECESSITY_CEILING
        ) == pytest.approx(1.0, abs=1e-6)
        assert grid.floor_pass_probability(
            MARGINS, study.REPLICATES, float(row["coverage"]), study.NECESSITY_FLOOR
        ) == pytest.approx(1.0, abs=1e-6)
        assert grid.ceiling_pass_probability(
            MARGINS, study.REPLICATES, 0.87, study.NECESSITY_CEILING
        ) == pytest.approx(1.0, abs=1e-6)

    def test_the_reference_at_k1_covers_near_nominal(self, summary: pd.DataFrame) -> None:
        row = reading(summary, "L4", "mechanism_correct_sv_one_step_K1", laws.L4_RISK)
        assert row["plain_coverage"] > 0.92 and row["coverage"] > 0.92


class TestTheNoiseArm:
    @pytest.mark.parametrize("label", study.CALIBRATION_LABELS)
    def test_the_noise_scale_is_the_pilot_spread(self, summary: pd.DataFrame, label: str) -> None:
        spread = float(label_reading(summary, label)["empirical_sd"]) * np.sqrt(laws.N)
        assert properties.NOISE_SD[label] == pytest.approx(spread, rel=5e-3)


class TestTheFits:
    def test_the_threshold_patch_reaches_the_hazard_walk(self) -> None:
        frame, _ = laws.draw_l4(laws.N, 5)
        original = score_module.rule_thresholds
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaled = laws.fit_l4(
                frame,
                stopping="scaled_var",
                configuration="mechanism_correct",
                scale=laws.NECESSITY_SCALE,
            )
            plain = laws.fit_l4(frame, stopping="scaled_var", configuration="mechanism_correct")
        assert score_module.rule_thresholds is original
        b8 = abs(scaled[laws.L4_RISK].score) / scaled[laws.L4_RISK].std_error
        b1 = abs(plain[laws.L4_RISK].score) / plain[laws.L4_RISK].std_error
        assert b8 > 2.0 * b1
        assert scaled[laws.L4_RISK].rule_widened and plain[laws.L4_RISK].rule_widened


class TestTheRedCellReading:
    """The declared red-cell reading, applied by code to the committed tables.

    The gated cells are every ``interval_calibration`` cell (the positives and their derived
    controls), the gated W cells and the N pair; the ``cleverly`` truth rows gate too.  Without
    committed tables the result is empty, so the test bites at registration.
    """

    @staticmethod
    def gating_cells() -> tuple[str, ...]:
        roles = {(c.family, c.name): c.role for c in study.CELLS}
        calibration = tuple(
            f"interval_calibration/{cell}" for cell in study.PROPERTY_CELLS["interval_calibration"]
        )
        witnesses = tuple(
            f"{family}/{name}"
            for (family, name), role in roles.items()
            if family == "residual_witness" and role == "positive"
        )
        necessity = tuple(
            f"residual_necessity/{cell}" for cell in study.PROPERTY_CELLS["residual_necessity"]
        )
        return calibration + witnesses + necessity

    def test_the_gating_cells_are_declared(self) -> None:
        declared = {f"{f}/{c}" for f, cells in study.PROPERTY_CELLS.items() for c in cells}
        cells = self.gating_cells()
        assert set(cells) <= declared
        assert len(cells) == len(set(cells)) == GATING_COUNT

    def test_no_red_gating_row_lacks_an_admissible_reading(self) -> None:
        assert tools.gating_findings(study.STUDY, self.gating_cells(), ("cleverly",)) == []
