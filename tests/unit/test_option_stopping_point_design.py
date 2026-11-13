"""The design of ``option-stopping-point`` (Fix-8 plan sections 5.0 and 5.1), recomputed.

Every declared number of :mod:`tests.studies.canonical_option_stopping_point` and
:mod:`tests.studies.option_stopping_point_properties` is computed here, from the exact laws and
from the committed pilot (:mod:`tests.studies.option_stopping_pilot`, 2,000 draws per cell):

* the exact truths of L1, L2 and LN, and LN's exact-law initial bias and predicted plain
  coverage at ``K = 8``;
* the pre-read: every gated calibration cell reads pilot coverage of at least 0.937, and every
  gated witness cell a pilot share of at least 0.7;
* the replications: the product floor and the arm-pooled predictive rule;
* the pass probabilities of every control and of W and N at the pilot's readings;
* the noise arm's scale against the pilot's spread;
* that the threshold patch reaches the fit, and the column order LN's outcome learner reads.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import cleverly.fluctuation._score as score_module
from tests.studies import _option_stopping_cells as grid
from tests.studies import _option_stopping_laws as laws
from tests.studies import canonical_option_stopping_point as study
from tests.studies import canonical_tmle
from tests.studies import mtp_study_tools as tools
from tests.studies import option_stopping_pilot as pilot
from tests.studies import option_stopping_point_properties as properties
from tests.studies import repeated_sizing as sizing

MARGINS = study.STUDY.margins
#: The pre-read floor: below it the band gate passes with probability below about one half.
PRE_READ_COVERAGE = 0.937
#: The pilot share under which a W cell is declared reported instead of gated (plan 5.1).
PRE_READ_WITNESS = 0.7
#: 24 calibration cells, 2 gated W cells and the N pair.
GATING_COUNT = 28


@pytest.fixture(scope="module")
def summary() -> pd.DataFrame:
    return pilot.read_summary()


def reading(summary: pd.DataFrame, law: str, cell: str, parameter: str) -> pd.Series:
    rows = summary.loc[
        (summary["study"] == "point")
        & (summary["law"] == law)
        & (summary["cell"] == cell)
        & (summary["parameter"] == parameter)
    ]
    assert len(rows) == 1, (law, cell, parameter)
    return rows.iloc[0]


def calibration_cells() -> list[grid.Cell]:
    return [c for c in study.CELLS if c.family == "interval_calibration"]


class TestTheLaws:
    @pytest.mark.parametrize("outcome", study.OUTCOMES)
    def test_l1_truths_are_canonical_tmle_s(self, outcome: str) -> None:
        exact = canonical_tmle.truth_for(canonical_tmle.scenario_dgp(outcome))
        assert study.TRUTHS[outcome]["ate"] == pytest.approx(exact["ate"], abs=1e-15)
        _, drawn = laws.draw_l1(outcome, 20, 1)
        assert drawn["ate"] == pytest.approx(study.TRUTHS[outcome]["ate"], abs=1e-15)

    def test_ln_ey1_is_point_one_exactly(self) -> None:
        w1, w2, mass = laws._ln_grid()
        assert float(mass @ laws.ln_outcome_mean(1.0, w1, w2)) == pytest.approx(0.1, abs=1e-12)
        assert study.TRUTHS["LN"]["ey1"] == 0.1

    def test_l2_truths(self) -> None:
        assert study.TRUTHS["L2"] == {"ey1": 0.1, "ate": 0.1}

    def test_the_ln_initial_fit_is_biased_by_a_fixed_amount(self) -> None:
        limit = laws.ln_initial_limit()
        assert limit["psi_initial"] == pytest.approx(0.41805, abs=1e-5)
        # The initial score in standard errors at n = 2,000 is far outside K c_n, so the walk
        # starts outside the scaled threshold and stops near it.
        standardized = limit["initial_bias"] / (limit["initial_sd"] / np.sqrt(laws.N))
        assert standardized > 2.5 > laws.NECESSITY_SCALE * laws.unit_cap(laws.N)

    def test_the_predicted_plain_coverage_at_k8(self, summary: pd.DataFrame) -> None:
        # At the scaled rule's cap b = K c_n the plain interval's coverage is
        # Phi(z - b) - Phi(-z - b), its least value over residuals the met rule allows.
        cap = laws.NECESSITY_SCALE * laws.unit_cap(laws.N)
        worst = laws.predicted_plain_coverage(cap, pilot.CRITICAL)
        assert worst == pytest.approx(0.817, abs=2e-3)
        observed = reading(summary, "LN", "sv_one_step_K8", "ey1")
        # The walk approaches the scaled threshold from the initial score's side and stops just
        # inside it, so the residual is nearly the constant b, one-signed, below the cap.
        b = float(observed["mean_abs_residual"])
        assert abs(float(observed["mean_residual"])) == pytest.approx(b, rel=1e-12)
        assert 0.85 * cap < b <= cap
        # The pilot's plain coverage sits at the prediction at that b (3 binomial SDs), and
        # above the worst case at the cap.
        predicted = laws.predicted_plain_coverage(b, pilot.CRITICAL)
        spread = np.sqrt(predicted * (1 - predicted) / observed["draws"])
        assert abs(observed["plain_coverage"] - predicted) < 3.0 * spread
        assert observed["plain_coverage"] > worst


class TestTheDeclaration:
    def test_the_constants(self) -> None:
        assert study.PRIMARY_N == laws.N == 2_000
        assert (laws.TOL, laws.MAX_ITER, laws.ONE_STEP_MAX_ITER) == (1e-10, 100, 200)
        assert laws.NECESSITY_SCALE == 8.0
        assert (study.WITNESS_SHARE, study.NECESSITY_CEILING, study.NECESSITY_FLOOR) == (
            0.5,
            0.92,
            0.90,
        )
        assert study.STEP_BOUND == 0.05
        assert grid.WITNESS_FLOOR == pilot.WITNESS_FLOOR == 0.01
        assert study.STUDY.publication_policy == "reporting"
        assert study.STUDY.reference is None

    def test_the_seeds_are_the_study_s_own(self) -> None:
        from tests.studies import canonical_option_stopping_ltmle as sibling
        from tests.studies.evidence.registry import registered

        seeds = {record.seed for record in registered()} | {sibling.SEED, pilot.PILOT_SEED}
        assert study.SEED not in seeds
        assert study.RESAMPLING_SEED not in {r.resampling_seed for r in registered()}

    def test_the_record_declares_the_cells_the_module_builds(self) -> None:
        assert study.STUDY.property_cells == properties.PROPERTY_CELLS
        built = {(c.property, c.cell) for c in properties.declared_cells()}
        declared = {(f, c) for f, cells in study.PROPERTY_CELLS.items() for c in cells}
        assert built == declared

    def test_eight_gated_calibration_positives_with_two_controls_each(self) -> None:
        positives = [c for c in calibration_cells() if c.role == "positive"]
        assert len(positives) == 8
        assert {c.estimand for c in positives} == {"ate"}
        assert len(study.PROPERTY_CELLS["interval_calibration"]) == 24
        assert set(properties.DERIVED_CELLS) == {
            f"interval_calibration/{c.name}" for c in positives
        }

    def test_the_gated_witness_and_necessity_cells(self) -> None:
        roles = {(c.family, c.name): c.role for c in study.CELLS}
        gated = {
            name
            for (family, name), role in roles.items()
            if family == "residual_witness" and role == "positive"
        }
        assert gated == {"binary_sv_one_step__witness", "continuous_sv_one_step__witness"}
        assert roles[("residual_necessity", "ln_ey1__widened")] == "positive"
        assert roles[("residual_necessity", "ln_ey1__plain")] == "control"
        reported = {c.family for c in study.CELLS if c.role == "diagnostic"}
        assert reported - {"residual_witness"} == set(study.REPORTED_FAMILIES)

    def test_every_rule_fit_records_its_rule_and_residual(self) -> None:
        assert grid.ROW_COLUMNS == ("residual", "rule_met")


class TestThePreRead:
    def test_the_pilot_ran_at_least_2000_draws_per_cell(self, summary: pd.DataFrame) -> None:
        assert (summary.loc[summary["study"] == "point", "draws"] >= pilot.DRAWS).all()

    @pytest.mark.parametrize("cell", [c.name for c in calibration_cells() if c.role == "positive"])
    def test_every_gated_calibration_cell_reads_coverage_above_the_pre_read_floor(
        self, summary: pd.DataFrame, cell: str
    ) -> None:
        label = cell.split("__")[0]
        law, fit = label.split("_", 1)
        row = reading(summary, law, fit, "ate")
        assert row["coverage"] >= PRE_READ_COVERAGE
        assert (
            sizing.calibration_pass(MARGINS, study.REPLICATES, np.array([row["se_ratio"]]))[0] > 0.5
        )

    @pytest.mark.parametrize("outcome", study.OUTCOMES)
    def test_every_gated_witness_cell_reads_a_share_of_at_least_0_7(
        self, summary: pd.DataFrame, outcome: str
    ) -> None:
        row = reading(summary, outcome, "sv_one_step", "ate")
        assert row["residual_share"] >= PRE_READ_WITNESS

    def test_the_rule_cells_meet_their_rule(self, summary: pd.DataFrame) -> None:
        rows = summary.loc[(summary["study"] == "point") & (summary["law"] != "LN")]
        assert (rows["rule_met_share"] >= 0.99).all()


class TestTheSizes:
    def test_the_product_floor(self) -> None:
        def product(replicates: int) -> float:
            cell = float(sizing.calibration_pass(MARGINS, replicates, np.array([1.0]))[0])
            return cell**8

        floor = next(r for r in range(400, 20_001, 400) if product(r) >= 0.95)
        assert floor == 3_200
        assert product(floor) == pytest.approx(0.980, abs=1e-3)

    def test_the_pooled_predictive_rule_sets_the_replications(self, summary: pd.DataFrame) -> None:
        ratios = {}
        for cell in calibration_cells():
            if cell.role != "positive":
                continue
            law, fit = cell.label.split("_", 1)
            ratios[cell.name] = float(reading(summary, law, fit, "ate")["se_ratio"])

        def correlations(labels: Any) -> np.ndarray:
            # Cells on one law read the same draws, so their estimates are taken as perfectly
            # correlated (pilot correlation at least 0.998); the two laws share none.
            laws_of = [label.split("_", 1)[0] for label in labels]
            return np.array([[float(a == b) for b in laws_of] for a in laws_of])

        pooled = sizing.calibration_replicates(MARGINS, ratios, pilot.DRAWS, correlations)
        assert max(3_200, pooled) == study.REPLICATES

    def test_the_derived_controls_fail_with_certainty(self) -> None:
        for ratio in (properties.SHRUNKEN_SE_FACTOR, 1.0 / np.sqrt(2.0)):
            assert sizing.calibration_control_probability(
                MARGINS, study.REPLICATES, ratio
            ) == pytest.approx(1.0, abs=1e-6)

    def test_the_witness_passes_at_the_pilot_share(self, summary: pd.DataFrame) -> None:
        for outcome in study.OUTCOMES:
            share = float(reading(summary, outcome, "sv_one_step", "ate")["residual_share"])
            assert grid.floor_pass_probability(
                MARGINS, study.REPLICATES, share, study.WITNESS_SHARE
            ) == pytest.approx(1.0, abs=1e-6)
        # And at the declared pre-read floor itself.
        assert grid.floor_pass_probability(
            MARGINS, study.REPLICATES, PRE_READ_WITNESS, study.WITNESS_SHARE
        ) == pytest.approx(1.0, abs=1e-6)

    def test_the_necessity_pair_passes_at_the_pilot_readings(self, summary: pd.DataFrame) -> None:
        row = reading(summary, "LN", "sv_one_step_K8", "ey1")
        assert grid.ceiling_pass_probability(
            MARGINS, study.REPLICATES, float(row["plain_coverage"]), study.NECESSITY_CEILING
        ) == pytest.approx(1.0, abs=1e-6)
        assert grid.floor_pass_probability(
            MARGINS, study.REPLICATES, float(row["coverage"]), study.NECESSITY_FLOOR
        ) == pytest.approx(1.0, abs=1e-6)
        # The control at the plan's stated true coverage of 0.86.
        assert grid.ceiling_pass_probability(
            MARGINS, study.REPLICATES, 0.86, study.NECESSITY_CEILING
        ) == pytest.approx(1.0, abs=1e-6)

    def test_the_reference_at_k1_covers_near_nominal(self, summary: pd.DataFrame) -> None:
        row = reading(summary, "LN", "sv_one_step_K1", "ey1")
        assert row["plain_coverage"] > 0.92 and row["coverage"] > 0.92


class TestTheNoiseArm:
    @pytest.mark.parametrize("label", study.CALIBRATION_LABELS)
    def test_the_noise_scale_is_the_pilot_spread(self, summary: pd.DataFrame, label: str) -> None:
        law, fit = label.split("_", 1)
        row = reading(summary, law, fit, "ate")
        spread = float(row["empirical_sd"]) * np.sqrt(laws.N)
        assert properties.NOISE_SD[label] == pytest.approx(spread, rel=5e-3)


class TestTheFits:
    def test_the_threshold_patch_reaches_the_fit(self) -> None:
        frame, _ = laws.draw_ln(laws.N, 7)
        original = score_module.rule_thresholds
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaled = laws.fit_point(
                "LN",
                frame,
                stopping="scaled_var",
                targeting="one_step",
                estimands=("ey1",),
                scale=laws.NECESSITY_SCALE,
            )["ey1"]
            plain = laws.fit_point(
                "LN", frame, stopping="scaled_var", targeting="one_step", estimands=("ey1",)
            )["ey1"]
        assert score_module.rule_thresholds is original
        cap = laws.unit_cap(laws.N)
        b8 = abs(scaled.score) / scaled.std_error
        b1 = abs(plain.score) / plain.std_error
        # The scaled walk stops near K c_n, outside the unscaled threshold.
        assert cap < b8 <= 1.05 * laws.NECESSITY_SCALE * cap
        assert b8 > 0.5 * laws.NECESSITY_SCALE * cap
        assert b1 <= 1.05 * cap

    def test_the_ln_outcome_learner_reads_a_w1_w2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        frame, _ = laws.draw_ln(400, 3)
        seen: list[np.ndarray] = []
        fit = LinearRegression.fit

        def record(self: LinearRegression, x: Any, y: Any, *args: Any, **kwargs: Any) -> Any:
            seen.append(np.asarray(x))
            return fit(self, x, y, *args, **kwargs)

        monkeypatch.setattr(LinearRegression, "fit", record)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            laws.fit_point("LN", frame, stopping="relative", estimands=("ey1",))
        assert seen
        expected = frame[["A", "W1", "W2"]].to_numpy()
        assert seen[0].shape[1] == 3
        np.testing.assert_allclose(seen[0], expected, rtol=0, atol=1e-12)

    def test_the_step_bound_cell_is_keyed_apart(self) -> None:
        fit = study.STREAMS["binary"].fits["ss_newton_step_bound"]
        assert fit.keywords["step_bound"] == study.STEP_BOUND
        assert fit.keywords["stopping"] == "sample_size"


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
