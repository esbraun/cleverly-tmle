"""The design of ``option-outcome-by-arm`` (Fix-8 plan sections 5.0 and 5.3), recomputed.

Every declared number of :mod:`tests.studies.canonical_option_outcome_by_arm` and
:mod:`tests.studies.option_outcome_by_arm_properties` is checked here: the exact truths, the
exact-law limits of every positive, control and reported cell with a fixed outcome scale, the
efficiency bounds that size the noise arms, the replication counts and their pass
probabilities, the pre-read of the committed pilot
(``tests/studies/option_outcome_by_arm_pilot.csv``), and the declared red-cell reading.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from tests.studies import _option_by_arm_law as law
from tests.studies import canonical_option_outcome_by_arm as study
from tests.studies import option_outcome_by_arm_pilot as pilot
from tests.studies import option_outcome_by_arm_properties as properties
from tests.studies import repeated_sizing as sizing
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.registry import Margins, registered

MARGINS = Margins()
#: Plan 5.0: a gated calibration cell whose pilot coverage reads below this is diagnosed first.
PRE_READ_COVERAGE = 0.937
#: Plan 5.0: a bias positive whose pilot bias exceeds this raises its replications.
PILOT_BIAS_CEILING = 0.12
#: Plan 5.0: a control's discrimination power at its exact-law limit.
CONTROL_POWER = 0.9999
#: Plan 5.0's product floor of the four calibration cells.
PRODUCT_FLOOR = 2_800

#: Each limit cell's setting.
LIMIT_SETTINGS = {
    ("double_robustness", "treatment_correct"): "q_intercept",
    ("double_robustness", "both_wrong"): "both_intercept",
    ("pooled_outcome", "in_sample_binary"): "pooled_binary_in_sample",
    ("pooled_outcome", "stacked_binary"): "pooled_binary_stacked",
    ("pooled_outcome", "stacked_bounded"): "pooled_bounded_stacked",
}


@functools.cache
def limit(name: str, n: int = 2_000) -> tuple[float, float]:
    """``(bias over SD at n, reported SE over SD)`` of a setting with a fixed outcome scale."""
    variance = law.stacked_variance(name)
    bias = law.setting_limit(name).ate - law.truth(law.SETTINGS[name].law)["ate"]
    return bias / variance.sd(n), variance.se_ratio


@pytest.fixture(scope="module")
def summary() -> pd.DataFrame:
    return pd.read_csv(pilot.SUMMARY)


def reading(summary: pd.DataFrame, cell: str) -> pd.Series:
    rows = summary.loc[summary["cell"] == cell]
    assert len(rows) == 1, cell
    return rows.iloc[0]


def crude_limit(n: int) -> float:
    """The difference in arm means under ``LB`` minus the ATE, over its own SD at ``n``."""
    at = law.grid()
    g = law.g0(at.w1, at.w2)
    treated = at.mean(g)
    one = at.mean(g * law.qbar0("LB", 1.0, at.w1, at.w2)) / treated
    zero = at.mean((1.0 - g) * law.qbar0("LB", 0.0, at.w1, at.w2)) / (1.0 - treated)
    variance = one * (1.0 - one) / treated + zero * (1.0 - zero) / (1.0 - treated)
    return (one - zero - law.truth("LB")["ate"]) / np.sqrt(variance / n)


class TestTheDeclaration:
    def test_the_constants(self) -> None:
        assert study.SLUG == "option-outcome-by-arm"
        assert (study.SEED, study.RESAMPLING_SEED) == (20267531, 20267532)
        assert study.STUDY.publication_policy == "reporting"
        assert (study.PRIMARY_N, study.PRIMARY_REPLICATES) == (2_000, 2_800)
        assert properties.ROBUSTNESS_REPLICATES == 1_600
        assert (law.G_BOUNDS, law.Q_BOUNDS, law.N_FOLDS) == ((0.01, 0.99), (0.0, 1.0), 5)
        assert (law.MAX_ITER, law.TOL) == (100, 1e-10)

    def test_the_seeds_are_the_study_s_own(self) -> None:
        seeds = {
            s
            for record in registered()
            if record.slug != study.SLUG
            for s in (record.seed, record.resampling_seed)
        }
        assert not {study.SEED, study.RESAMPLING_SEED, pilot.PILOT_SEED} & seeds

    def test_registered_with_its_committed_run(self) -> None:
        assert study.SLUG in {record.slug for record in registered()}
        assert study.STUDY.artifact("property-replicates.parquet").exists()
        from tests.unit.test_method_evidence import UNCOMMITTED_MODULES

        assert properties.__name__ not in UNCOMMITTED_MODULES

    def test_the_record_declares_the_cells_the_module_builds(self) -> None:
        built: dict[str, set[str]] = {}
        for cell in properties.declared_cells():
            built.setdefault(cell.property, set()).add(cell.cell)
        for derived in properties.DERIVED_CELLS.values():
            for name in derived:
                family, derived_cell = name.split("/", 1)
                built[family].add(derived_cell)
        assert built == {family: set(cells) for family, cells in study.PROPERTY_CELLS.items()}

    def test_the_roles(self) -> None:
        roles = {(c.property, c.cell): c.role for c in properties.declared_cells()}
        for label in properties.CALIBRATION_LABELS:
            assert roles[("interval_calibration", f"{label}__correctly_specified")] == "positive"
            assert roles[("pooled_outcome", label)] == property_verdicts.DIAGNOSTIC_ROLE
        assert roles[("double_robustness", "treatment_correct")] == "positive"
        assert roles[("double_robustness", "both_wrong")] == "control"

    def test_the_cells_fit_the_declared_option(self) -> None:
        for label, (by_arm, pooled, _) in properties.LABEL_SETTINGS.items():
            assert law.SETTINGS[by_arm].by_arm and not law.SETTINGS[pooled].by_arm
            assert law.SETTINGS[by_arm].law == law.SETTINGS[pooled].law
            assert law.SETTINGS[by_arm].cross_fit == law.SETTINGS[pooled].cross_fit
            assert law.SETTINGS[by_arm].cross_fit == label.startswith("stacked")
        assert law.SETTINGS["q_intercept"].by_arm and law.SETTINGS["both_intercept"].by_arm

    def test_paired_cells_share_their_draws(self) -> None:
        seeds: dict[tuple[str, str], list[int]] = {}
        for ((family, cell, _, _, _, seed, _),) in properties.payloads(budget=2):
            seeds.setdefault((family, cell), []).append(seed)
        for label in properties.LABEL_SETTINGS:
            assert (
                seeds[("interval_calibration", f"{label}__correctly_specified")]
                == seeds[("pooled_outcome", label)]
            )
        assert (
            seeds[("interval_calibration", "in_sample_binary__correctly_specified")]
            == seeds[("interval_calibration", "stacked_binary__correctly_specified")]
        )
        # Three calibration streams and the two robustness cells' own.
        assert len({tuple(values) for values in seeds.values()}) == 5


class TestTheLaws:
    @pytest.mark.parametrize("name", law.LAWS)
    def test_the_truths_converge_and_match_monte_carlo(self, name: str) -> None:
        exact = law.truth(name)
        assert law.truth(name, 120)["ate"] == pytest.approx(exact["ate"], abs=1e-12)
        w1, w2 = np.random.default_rng(7).standard_normal((2, 2_000_000))
        monte = np.mean(law.qbar0(name, 1.0, w1, w2) - law.qbar0(name, 0.0, w1, w2))
        assert monte == pytest.approx(exact["ate"], abs=5e-4)

    def test_the_closed_forms(self) -> None:
        assert law.truth("LB-c")["ate"] == pytest.approx(0.3, abs=1e-12)
        assert law.truth("LB-b") == pytest.approx(law.truth("LB"), abs=1e-15)

    def test_the_draws_have_the_law_s_moments(self) -> None:
        for name in law.LAWS:
            frame = law.draw(name, 400_000, 3)
            for a in (1.0, 0.0):
                rows = frame["A"] == a
                w1, w2 = frame.loc[rows, "W1"], frame.loc[rows, "W2"]
                residual = frame.loc[rows, "Y"] - law.qbar0(name, a, w1, w2)
                assert abs(residual.mean()) < 5e-3
                variance = law.second_moment(name, a, w1, w2) - law.qbar0(name, a, w1, w2) ** 2
                assert (residual**2).mean() == pytest.approx(variance.mean(), rel=0.02)

    @pytest.mark.parametrize("name", ["binary_in_sample", "bounded_stacked"])
    def test_the_per_arm_glm_is_the_law(self, name: str) -> None:
        beta = law.outcome_coefficients(name)
        assert beta == pytest.approx([*law.ARM_INDEX[1.0], *law.ARM_INDEX[0.0]], abs=1e-9)
        assert law.mechanism_coefficients(name) == pytest.approx(law.TREATMENT, abs=1e-9)

    def test_the_pooled_glm_misses_the_interaction(self) -> None:
        at = law.grid()
        q = law.outcome_limit("pooled_binary_in_sample")
        g = law.g0(at.w1, at.w2)
        error = (
            g * (q(1.0) - law.qbar0("LB", 1.0, at.w1, at.w2)) ** 2
            + (1.0 - g) * (q(0.0) - law.qbar0("LB", 0.0, at.w1, at.w2)) ** 2
        )
        assert at.mean(error) > 1e-3

    def test_the_bounds_rarely_bind(self) -> None:
        at = law.grid()
        g = law.g0(at.w1, at.w2)
        assert at.mean((g < law.G_BOUNDS[0]) | (g > law.G_BOUNDS[1])) < 1e-6

    def test_the_efficiency_bounds_size_the_noise_arms(self) -> None:
        assert set(properties.EFFICIENCY_SD) == set(properties.CALIBRATION_LABELS)
        for label, value in properties.EFFICIENCY_SD.items():
            setting = properties.LABEL_SETTINGS[label][0]
            assert value == pytest.approx(law.efficiency_sd(law.SETTINGS[setting].law), abs=1e-6)

    @pytest.mark.parametrize("name", ["binary_in_sample", "bounded_stacked"])
    def test_the_stacked_variance_of_a_correct_fit_is_the_bound(self, name: str) -> None:
        """A witness of the stacked-equation computation: correct fits reach ``E[D*^2]``."""
        variance = law.stacked_variance(name)
        bound = law.efficiency_sd(law.SETTINGS[name].law)
        assert np.sqrt(variance.true) == pytest.approx(bound, rel=1e-6)
        assert variance.se_ratio == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize(
        "name", ["pooled_binary_in_sample", "pooled_bounded_stacked", "q_intercept"]
    )
    def test_the_stacked_equations_are_the_estimator_s_off_the_truth(self, name: str) -> None:
        """``E[U(theta_0)] = 0`` where the outcome regression is wrong, so no term vanishes."""
        assert law.stacked_variance(name).residual < 1e-10

    def test_the_crude_variance_matches_its_closed_form(self) -> None:
        bias, _ = limit("both_intercept")
        assert bias == pytest.approx(crude_limit(2_000), rel=1e-6)


class TestTheExactLimits:
    @pytest.mark.parametrize("key", sorted(properties.EXACT_LIMITS))
    def test_the_declared_limits(self, key: tuple[str, str]) -> None:
        assert limit(LIMIT_SETTINGS[key]) == pytest.approx(properties.EXACT_LIMITS[key], abs=1e-4)

    @pytest.mark.parametrize("name", ["binary_in_sample", "binary_stacked", "bounded_stacked"])
    def test_every_calibration_positive_is_unbiased_and_calibrated(self, name: str) -> None:
        bias, ratio = limit(name)
        assert abs(bias) < 1e-10
        assert ratio == pytest.approx(1.0, abs=1e-6)

    def test_the_double_robustness_positive_is_unbiased(self) -> None:
        assert abs(law.setting_limit("q_intercept").ate - law.truth("LB")["ate"]) < 1e-12

    def test_the_control_reaches_the_floor(self) -> None:
        bias, _ = limit("both_intercept")
        assert bias >= law.CONTROL_FLOOR_SD

    def test_the_plan_s_mechanism_sign_left_the_control_below_the_floor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Plan 5.3's ``0.4 W1 - 0.4 W2``: the two arms' confounding cancels."""
        monkeypatch.setattr(law, "TREATMENT", (0.0, 0.4, -0.4))
        assert crude_limit(2_000) < 0.2

    def test_the_pooled_cells_are_unbiased_by_double_robustness(self) -> None:
        for name in ("pooled_binary_in_sample", "pooled_bounded_stacked"):
            assert (
                abs(law.setting_limit(name).ate - law.truth(law.SETTINGS[name].law)["ate"]) < 1e-12
            )


class TestTheSizes:
    def test_the_product_floor_of_four_calibration_cells(self) -> None:
        def product(replicates: int) -> float:
            return float(sizing.calibration_pass(MARGINS, replicates, np.array([1.0]))[0]) ** 4

        floor = next(r for r in range(400, 20_001, 400) if product(r) >= 0.95)
        assert floor == PRODUCT_FLOOR
        assert product(floor) == pytest.approx(0.969, abs=2e-3)

    def test_the_pooled_predictive_rule(self, summary: pd.DataFrame) -> None:
        ratios = {
            f"{label}__correctly_specified": float(
                reading(summary, properties.LABEL_SETTINGS[label][0])["se_ratio"]
            )
            for label in properties.CALIBRATION_LABELS
        }
        rho = float(reading(summary, "binary_in_sample")["estimate_correlation"])
        binary = {"in_sample_binary", "stacked_binary"}

        def correlations(labels: Sequence[str]) -> np.ndarray:
            return np.array(
                [
                    [1.0 if a == b else (rho if {a, b} <= binary else 0.0) for b in labels]
                    for a in labels
                ]
            )

        pooled = sizing.calibration_replicates(
            MARGINS, ratios, pilot.CALIBRATION_DRAWS, correlations
        )
        assert max(PRODUCT_FLOOR, pooled) == properties.CALIBRATION_REPLICATES

    def test_the_derived_controls_fail_with_certainty(self) -> None:
        r = properties.CALIBRATION_REPLICATES
        for ratio in (properties.SHRUNKEN_SE_FACTOR, 1.0 / np.sqrt(2.0)):
            assert sizing.calibration_control_probability(MARGINS, r, ratio) > CONTROL_POWER

    def test_the_control_is_discriminated(self) -> None:
        bias, _ = limit("both_intercept")
        power = sizing.bias_discriminated_probability(
            MARGINS, bias, properties.ROBUSTNESS_REPLICATES
        )
        assert power > CONTROL_POWER

    def test_the_bias_positive_passes_at_its_limit(self) -> None:
        probability = sizing.bias_equivalent_probability(
            MARGINS, 0.0, properties.ROBUSTNESS_REPLICATES
        )
        assert probability > 0.9997


class TestThePreRead:
    def test_the_calibration_pilot_is_the_shared_pilot_size(self) -> None:
        assert pilot.CALIBRATION_DRAWS == sizing.CALIBRATION_PILOT_DRAWS

    def test_the_pilot_ran_its_draws_from_draw_zero(self, summary: pd.DataFrame) -> None:
        for block, (_, _, cells) in pilot.BLOCKS.items():
            draws = pilot.CALIBRATION_DRAWS if block in pilot.CALIBRATION_BLOCKS else pilot.DRAWS
            for cell in cells:
                row = reading(summary, cell)
                assert (int(row["draws"]), int(row["first_draw"])) == (draws, 0)
                assert int(row["last_draw"]) == draws - 1

    @pytest.mark.parametrize("label", properties.CALIBRATION_LABELS)
    def test_each_gated_calibration_cell_reads_above_the_pre_read_floor(
        self, summary: pd.DataFrame, label: str
    ) -> None:
        row = reading(summary, properties.LABEL_SETTINGS[label][0])
        assert row["coverage"] >= PRE_READ_COVERAGE
        replicates = properties.CALIBRATION_REPLICATES
        assert sizing.calibration_pass(MARGINS, replicates, np.array([row["se_ratio"]]))[0] > 0.5

    def test_the_bias_positive_reads_below_the_ceiling(self, summary: pd.DataFrame) -> None:
        row = reading(summary, "q_intercept")
        assert abs(row["bias_over_sd"]) <= PILOT_BIAS_CEILING
        probability = sizing.bias_equivalent_probability(
            MARGINS, float(row["bias_over_sd"]), properties.ROBUSTNESS_REPLICATES
        )
        assert probability > 0.99
        low, high = property_verdicts.UNION_MODEL_SE_BAND
        assert low < row["se_ratio"] < high

    def test_the_control_reads_near_its_limit(self, summary: pd.DataFrame) -> None:
        row = reading(summary, "both_intercept")
        assert row["bias_over_sd"] == pytest.approx(limit("both_intercept")[0], abs=0.15)

    @pytest.mark.parametrize("key", sorted(LIMIT_SETTINGS))
    def test_the_limit_cells_read_near_their_se_ratio(
        self, summary: pd.DataFrame, key: tuple[str, str]
    ) -> None:
        name = LIMIT_SETTINGS[key]
        row = reading(summary, name)
        draws = int(row["draws"])
        # Five Monte Carlo errors of an SE ratio, 1 / sqrt(2 (R - 1)) relative.
        assert row["se_ratio"] == pytest.approx(limit(name)[1], rel=5 / np.sqrt(2 * (draws - 1)))


#: The cells whose red verdict gates the study: the calibration positives, their derived
#: controls and the double-robustness pair.
GATED = (
    *(f"interval_calibration/{label}__correctly_specified" for label in study.CALIBRATION_LABELS),
    *(name for derived in properties.DERIVED_CELLS.values() for name in derived),
    "double_robustness/treatment_correct",
    "double_robustness/both_wrong",
)


class TestTheRedCellReading:
    def test_the_gated_cells_are_every_cell_that_is_not_reported(self) -> None:
        declared = {
            f"{family}/{cell}"
            for family, cells in study.PROPERTY_CELLS.items()
            if family not in properties.REPORTED_FAMILIES
            for cell in cells
        }
        assert len(GATED) == 14
        assert set(GATED) == declared

    def test_no_gated_cell_reads_an_inadmissible_red(self) -> None:
        """The declared red-cell reading, applied mechanically.

        Empty until the run commits its tables; at registration a red control, or a red
        positive without its finite-sample owner, fails here.
        """
        from tests.studies.mtp_study_tools import gating_findings

        assert gating_findings(study.STUDY, GATED, truth_implementations=("cleverly",)) == []
