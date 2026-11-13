"""The design of ``option-treatment-adjustment`` (Fix-8 plan sections 5.0 and 5.4), recomputed.

Every declared number of :mod:`tests.studies.canonical_option_treatment_adjustment` and
:mod:`tests.studies.option_treatment_adjustment_properties` is checked here: the exact truths,
the exact-law limits of every instrument cell and control, the efficiency bound that sizes the
noise arm, the replication counts and their pass probabilities, and the pre-read of the
committed pilot (``tests/studies/option_treatment_adjustment_pilot.csv``).
"""

from __future__ import annotations

import functools
import itertools
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from tests.studies import _option_adjustment_laws as laws
from tests.studies import canonical_option_treatment_adjustment as study
from tests.studies import option_treatment_adjustment_pilot as pilot
from tests.studies import option_treatment_adjustment_properties as properties
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


@functools.cache
def limit(name: str, n: int) -> tuple[float, float]:
    """``(bias over SD, reported SE over SD)`` of a GLM setting at ``n``."""
    variance = laws.stacked_variance(name, n)
    bias = laws.setting_limit(name, n).ate - laws.truth(laws.SETTINGS[name].law)["ate"]
    return bias / variance.sd(n), variance.se_ratio


@pytest.fixture(scope="module")
def summary() -> pd.DataFrame:
    return pd.read_csv(pilot.SUMMARY)


def reading(summary: pd.DataFrame, cell: str) -> pd.Series:
    rows = summary.loc[summary["cell"] == cell]
    assert len(rows) == 1, cell
    return rows.iloc[0]


def naive_limit(n: int) -> float:
    """The intercept-only control's bias over its SD: the difference in arm means under LDA."""
    at = laws.grid()
    g = laws.g0("LDA", at.w1, at.w2, at.z)
    treated = at.mean(g)
    one = at.mean(g * laws.qbar0("LDA", 1.0, at.w1, at.w2, at.z)) / treated
    zero = at.mean((1.0 - g) * laws.qbar0("LDA", 0.0, at.w1, at.w2, at.z)) / (1.0 - treated)
    variance = one * (1.0 - one) / treated + zero * (1.0 - zero) / (1.0 - treated)
    return (one - zero - laws.truth("LDA")["ate"]) / np.sqrt(variance / n)


#: The one registered study that shares this record's seed.  Parts D and E were declared in
#: parallel and both took 20267641/20267642, and the committed runs drew from it.  Their primary
#: replicates read the same integer seeds on different laws (point ``LDA``, longitudinal
#: ``LE``), so the samples are dependent across the two studies but not identical, and no
#: claim of either study reads the other.  Every property stream hashes a label of its own
#: (``stream_seed``), so the property draws do not share seeds.  Changing the seed would need
#: both studies rerun.
SHARED_SEED_STUDY = "option-ratio-trim"


class TestTheDeclaration:
    def test_the_constants(self) -> None:
        assert study.SLUG == "option-treatment-adjustment"
        assert (study.SEED, study.RESAMPLING_SEED) == (20267641, 20267642)
        assert study.STUDY.publication_policy == "reporting"
        assert (study.PRIMARY_N, study.PRIMARY_REPLICATES) == (2_000, 2_400)
        assert properties.LADDER == (1_000, 4_000, 16_000)
        assert properties.ROBUSTNESS_REPLICATES == properties.LADDER_REPLICATES == 1_600
        assert properties.PARAMETRIC_REPLICATES == 2_400
        assert laws.G_BOUNDS == (0.01, 0.99)
        assert (laws.MAX_ITER, laws.TOL) == (100, 1e-10)

    def test_the_seeds_are_the_study_s_own(self) -> None:
        seeds = {
            (record.seed, record.resampling_seed)
            for record in registered()
            if record.slug not in {study.SLUG, SHARED_SEED_STUDY}
        }
        assert not {study.SEED, study.RESAMPLING_SEED} & {s for pair in seeds for s in pair}
        assert pilot.PILOT_SEED not in {s for pair in seeds for s in pair}
        shared = {r.seed for r in registered() if r.slug == SHARED_SEED_STUDY}
        assert shared == {study.SEED}

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
        built[properties.RATE_FAMILY] = {properties.RATE_CELL}
        assert built == {family: set(cells) for family, cells in study.PROPERTY_CELLS.items()}

    def test_the_roles(self) -> None:
        roles = {(c.property, c.cell): c.role for c in properties.declared_cells()}
        assert roles[("interval_calibration", "in_sample__correctly_specified")] == "positive"
        assert roles[("double_robustness", "treatment_correct")] == "positive"
        assert roles[("double_robustness", "both_wrong")] == "control"
        assert roles[("instrument_slow_outcome", "n16000")] == "control"
        reported = {
            role for (family, _), role in roles.items() if family in properties.REPORTED_FAMILIES
        }
        assert reported == {property_verdicts.DIAGNOSTIC_ROLE}

    def test_paired_cells_share_their_draws(self) -> None:
        keys = properties.payloads(budget=2)
        seeds: dict[tuple[str, str], list[int]] = {}
        for ((family, cell, _, _, _, seed, _),) in keys:
            seeds.setdefault((family, cell), []).append(seed)
        calibration = [
            seeds[("interval_calibration", f"{label}__correctly_specified")]
            for label in properties.CALIBRATION_LABELS
        ]
        assert calibration[0] == calibration[1]
        for n in properties.LADDER:
            family = "instrument_slow_outcome" if n == 16_000 else "instrument_slow_outcome_ladder"
            assert seeds[(family, f"n{n}")] == seeds[("instrument_reference", f"n{n}")]
        streams = {tuple(values) for values in seeds.values()}
        # Two shared pairs per ladder size and one calibration pair; every other cell its own.
        assert len(streams) == len(seeds) - 1 - len(properties.LADDER)


class TestTheLaws:
    @pytest.mark.parametrize("law", laws.LAWS)
    def test_the_truths_match_the_grid_and_monte_carlo(self, law: str) -> None:
        exact = laws.truth(law)
        at = laws.grid()
        grid = at.mean(laws.qbar0(law, 1.0, at.w1, at.w2, at.z)) - at.mean(
            laws.qbar0(law, 0.0, at.w1, at.w2, at.z)
        )
        assert grid == pytest.approx(exact["ate"], abs=1e-12)
        w1, w2, z = np.random.default_rng(7).standard_normal((3, 2_000_000))
        monte = np.mean(laws.qbar0(law, 1.0, w1, w2, z) - laws.qbar0(law, 0.0, w1, w2, z))
        assert monte == pytest.approx(exact["ate"], abs=5e-4)

    def test_the_lda_mechanism_glm_on_the_subset_is_correct(self) -> None:
        assert laws.mechanism_limit("LDA", laws.SUBSET) == pytest.approx([0, 0.5, -0.5], abs=1e-9)

    def test_the_ldb_mechanism_glm_on_the_subset_is_wrong(self) -> None:
        beta = laws.mechanism_limit("LDB", laws.SUBSET)
        at = laws.grid()
        fitted = expit(beta[0] + beta[1] * at.w1 + beta[2] * at.w2)
        true = laws.g0("LDB", at.w1, at.w2, at.z)
        assert at.mean((true - fitted) ** 2) > 0.02

    def test_the_bounds_never_bind_on_lda(self) -> None:
        at = laws.grid(20)
        g = laws.g0("LDA", at.w1, at.w2, at.z)
        assert at.mean((g < laws.G_BOUNDS[0]) | (g > laws.G_BOUNDS[1])) < 1e-6

    def test_the_efficiency_bound_sizes_the_noise_arm(self) -> None:
        bound = laws.efficiency_sd("LDA")
        assert set(properties.EFFICIENCY_SD) == set(properties.CALIBRATION_LABELS)
        for value in properties.EFFICIENCY_SD.values():
            assert value == pytest.approx(bound, abs=1e-6)

    def test_the_stacked_variance_of_a_correct_fit_is_the_bound(self) -> None:
        """A witness of the stacked-equation computation: under LDA it equals ``E[D*^2]``."""
        variance = laws.stacked_variance("lda_in_sample")
        assert np.sqrt(variance.true) == pytest.approx(laws.efficiency_sd("LDA"), rel=1e-6)
        assert variance.se_ratio == pytest.approx(1.0, abs=1e-6)

    def test_the_stacked_equations_are_the_estimator_s_at_a_nonzero_offset(self) -> None:
        """``E[U(theta_0)] = 0`` where the constructed error is on (review m1).

        At offset 0 the outcome score reads the same with or without the offset, so the bound
        check above is blind to it.  ``PerturbedLogistic`` fits its coefficients without the
        offset; a score that carried it would not be zero at the GLM limit.
        """
        assert laws.slow_offset(1_000) > 0.1
        assert laws.stacked_variance("ldb_slow", 1_000).residual < 1e-10
        assert laws.stacked_variance("ldb_parametric").residual < 1e-10

    def test_the_double_robustness_positive_is_unbiased_in_the_limit(self) -> None:
        """``lda_q_intercept``: a constant outcome fit and the correct ``W_g`` mechanism."""
        at = laws.grid()
        g_true = np.asarray(laws.g0("LDA", at.w1, at.w2, at.z))
        names = laws.SUBSET
        g_fit = laws.mechanism_at(laws.mechanism_limit("LDA", names, at), names, at)
        assert np.max(np.abs(g_fit - np.clip(g_true, *laws.G_BOUNDS))) < 1e-8
        mean_y = at.mean(
            g_true * laws.qbar0("LDA", 1.0, at.w1, at.w2, at.z)
            + (1.0 - g_true) * laws.qbar0("LDA", 0.0, at.w1, at.w2, at.z)
        )
        limit = laws.targeted_limit(
            at,
            g_true=g_true,
            g_fit=g_fit,
            q0=lambda arm: np.asarray(laws.qbar0("LDA", arm, at.w1, at.w2, at.z)),
            initial=lambda arm: np.full(len(at.weight), mean_y),
        )
        truth = laws.truth("LDA")
        assert abs(limit.means[1.0] - truth["ey1"]) < 1e-10
        assert abs(limit.means[0.0] - truth["ey0"]) < 1e-10
        assert abs(limit.ate - truth["ate"]) < 1e-10


class TestTheExactLimits:
    @pytest.mark.parametrize("key", sorted(properties.EXACT_LIMITS))
    def test_the_declared_limits(self, key: tuple[str, str]) -> None:
        family, cell = key
        n = int(cell.removeprefix("n"))
        name = {
            "instrument_slow_outcome": "ldb_slow",
            "instrument_slow_outcome_ladder": "ldb_slow",
            "instrument_reference": "ldb_slow_full",
            "instrument_parametric": "ldb_parametric",
        }[family]
        bias, ratio = limit(name, n)
        assert (bias, ratio) == pytest.approx(properties.EXACT_LIMITS[key], abs=1e-4)

    def test_the_control_reaches_the_floor_and_k_is_declared(self) -> None:
        bias, _ = limit("ldb_slow", 16_000)
        assert bias < 0.0
        assert abs(bias) >= laws.CONTROL_FLOOR_SD
        # The smallest rung reads beyond the margin too, which a quarter of k would not.
        assert abs(limit("ldb_slow", 1_000)[0]) > 3 * MARGINS.standardized_bias
        assert abs(bias) / 4 < laws.CONTROL_FLOOR_SD + 0.05

    def test_the_standardized_bias_grows_like_n_to_the_quarter(self) -> None:
        ratios = [limit("ldb_slow", n)[0] for n in properties.LADDER]
        for small, large in itertools.pairwise(ratios):
            assert large / small == pytest.approx(4**0.25, rel=0.02)

    def test_the_full_mechanism_removes_the_term(self) -> None:
        for n in properties.LADDER:
            assert abs(limit("ldb_slow_full", n)[0]) < 0.01

    def test_the_parametric_case_b_cell_is_unbiased_with_an_off_se(self) -> None:
        bias, ratio = limit("ldb_parametric", 2_000)
        assert abs(bias) < 1e-12
        assert ratio < 0.95

    def test_the_naive_control_reaches_the_floor(self) -> None:
        assert naive_limit(properties.ROBUSTNESS_N) >= laws.CONTROL_FLOOR_SD


class TestTheSizes:
    def test_the_product_floor_of_two_calibration_cells(self) -> None:
        def product(replicates: int) -> float:
            return float(sizing.calibration_pass(MARGINS, replicates, np.array([1.0]))[0]) ** 2

        floor = next(r for r in range(400, 20_001, 400) if product(r) >= 0.95)
        assert floor == 2_400
        assert product(floor) == pytest.approx(0.954, abs=2e-3)

    def test_the_pooled_predictive_rule(self, summary: pd.DataFrame) -> None:
        ratios = {
            f"{label}__correctly_specified": float(reading(summary, label)["se_ratio"])
            for label in properties.CALIBRATION_LABELS
        }
        rho = float(reading(summary, "in_sample")["estimate_correlation"])

        def correlations(labels: Sequence[str]) -> np.ndarray:
            return np.array([[1.0 if a == b else rho for b in labels] for a in labels])

        pooled = sizing.calibration_replicates(
            MARGINS, ratios, pilot.CALIBRATION_DRAWS, correlations
        )
        assert max(2_400, pooled) == properties.CALIBRATION_REPLICATES

    def test_the_derived_controls_fail_with_certainty(self) -> None:
        r = properties.CALIBRATION_REPLICATES
        for ratio in (properties.SHRUNKEN_SE_FACTOR, 1.0 / np.sqrt(2.0)):
            assert sizing.calibration_control_probability(MARGINS, r, ratio) > CONTROL_POWER

    def test_the_controls_are_discriminated(self) -> None:
        lr, _ = limit("ldb_slow", 16_000)
        for gap in (lr, naive_limit(properties.ROBUSTNESS_N)):
            power = sizing.bias_discriminated_probability(
                MARGINS, gap, properties.LADDER_REPLICATES
            )
            assert power > CONTROL_POWER


class TestThePreRead:
    def test_the_calibration_pilot_is_the_shared_pilot_size(self) -> None:
        assert pilot.CALIBRATION_DRAWS == sizing.CALIBRATION_PILOT_DRAWS

    def test_the_pilot_ran_its_draws(self, summary: pd.DataFrame) -> None:
        for block, (_, _, fits) in pilot.BLOCKS.items():
            draws = {"calibration": pilot.CALIBRATION_DRAWS}.get(
                block, pilot.LADDER_DRAWS if block.startswith("ladder") else pilot.DRAWS
            )
            for cell in fits:
                assert int(reading(summary, cell)["draws"]) >= draws

    @pytest.mark.parametrize("label", properties.CALIBRATION_LABELS)
    def test_each_gated_calibration_cell_reads_above_the_pre_read_floor(
        self, summary: pd.DataFrame, label: str
    ) -> None:
        row = reading(summary, label)
        assert row["coverage"] >= PRE_READ_COVERAGE
        replicates = properties.CALIBRATION_REPLICATES
        assert sizing.calibration_pass(MARGINS, replicates, np.array([row["se_ratio"]]))[0] > 0.5

    def test_the_bias_positive_reads_below_the_ceiling(self, summary: pd.DataFrame) -> None:
        row = reading(summary, "treatment_correct")
        assert abs(row["bias_over_sd"]) <= PILOT_BIAS_CEILING
        probability = sizing.bias_equivalent_probability(
            MARGINS, float(row["bias_over_sd"]), properties.ROBUSTNESS_REPLICATES
        )
        assert probability > 0.99
        low, high = property_verdicts.UNION_MODEL_SE_BAND
        assert low < row["se_ratio"] < high

    def test_the_controls_read_near_their_limits(self, summary: pd.DataFrame) -> None:
        naive = reading(summary, "both_wrong")
        assert naive["bias_over_sd"] == pytest.approx(naive_limit(2_000), abs=0.15)
        for n in properties.LADDER:
            row = reading(summary, f"slow_n{n}")
            assert row["bias_over_sd"] == pytest.approx(limit("ldb_slow", n)[0], abs=0.15)
            reference = reading(summary, f"reference_n{n}")
            assert abs(reference["bias_over_sd"]) < 0.15

    def test_the_parametric_cell_reads_its_exact_se_ratio(self, summary: pd.DataFrame) -> None:
        row = reading(summary, "parametric_n2000")
        assert row["se_ratio"] == pytest.approx(limit("ldb_parametric", 2_000)[1], abs=0.06)


#: The cells whose red verdict gates the study: the calibration positives and their derived
#: controls, the double-robustness pair and the slow-outcome control.
GATED = (
    "interval_calibration/in_sample__correctly_specified",
    "interval_calibration/stacked__correctly_specified",
    *(name for derived in properties.DERIVED_CELLS.values() for name in derived),
    "double_robustness/treatment_correct",
    "double_robustness/both_wrong",
    "instrument_slow_outcome/n16000",
)


class TestTheRedCellReading:
    def test_the_gated_cells_are_every_cell_that_is_not_reported(self) -> None:
        reported = {*properties.REPORTED_FAMILIES, properties.RATE_FAMILY}
        declared = {
            f"{family}/{cell}"
            for family, cells in study.PROPERTY_CELLS.items()
            if family not in reported
            for cell in cells
        }
        assert len(GATED) == 9
        assert set(GATED) == declared

    def test_no_gated_cell_reads_an_inadmissible_red(self) -> None:
        """The declared red-cell reading, applied mechanically (review S1).

        Empty until the run commits its tables; at registration a red control, or a red
        positive without its finite-sample owner, fails here.
        """
        from tests.studies.mtp_study_tools import gating_findings

        assert gating_findings(study.STUDY, GATED, truth_implementations=("cleverly",)) == []
