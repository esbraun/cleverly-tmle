"""The design of ``known-mechanism-sample-weights`` (Fix-16 plan section 7 with rev 2.1), recomputed.

Every declared number of :mod:`tests.studies.canonical_known_sample_weights` and
:mod:`tests.studies.known_sample_weights_properties` is checked here: the exact truths of both
laws (law SW's from :math:`\\bar Q_w` and from :math:`\\bar Q_0`, with a guard on a weight that
reads ``Y``), the tilt, the Kish effective sample sizes, the exact SDs that size the noise arms,
the control's exact-law bias and its power, the replication counts and their family-wise pass
probability from the committed pilot (``tests/studies/known_sample_weights_pilot.csv``), the
smoke gate, and the red-cell reading.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from tests.studies import canonical_known_sample_weights as study
from tests.studies import known_sample_weights_law as law
from tests.studies import known_sample_weights_pilot as pilot
from tests.studies import known_sample_weights_properties as properties
from tests.studies import repeated_sizing as sizing
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.registry import Margins, registered

MARGINS = Margins()
#: The family-wise pass probability the replication rule requires (guide "Sizing").
FAMILY_PASS = 0.95
#: Plan 7: the control's power at n = 4,000 and 16,000.
CONTROL_POWER = 0.9
#: The derived calibration controls must fail with at least this probability.
DERIVED_POWER = 0.9999
#: Plan rev 2.1 item 2: the longitudinal law's Kish effective sample size over n.
KISH_FLOOR = 0.5
#: How far law SW's truth must sit from the unweighted ATE, and the Y-reading guard's gap.
GUARD = 0.05
#: Monte Carlo draws of the truth cross-checks (plan 7: 10^7) and their tolerance (5 SE).
MC_DRAWS = 10_000_000
MC_SEED = 20261117


@functools.cache
def summary() -> pd.DataFrame:
    return pd.read_csv(pilot.SUMMARY)


def reading(block: str, cell: str) -> pd.Series:
    rows = summary().loc[(summary()["block"] == block) & (summary()["cell"] == cell)]
    assert len(rows) == 1, (block, cell)
    return rows.iloc[0]


def block_of(label: str) -> str:
    """The pilot block of a property label: ``sw_n1000`` reads ``sw_1000``."""
    return label.replace("_n", "_")


def gated_fit(label: str) -> str:
    return "ltmle_sample" if label == properties.LONGITUDINAL_LABEL else "sample_correct"


# ------------------------------------------------------------------------ the declaration
class TestTheDeclaration:
    def test_the_constants(self) -> None:
        assert study.SLUG == "known-mechanism-sample-weights"
        assert (study.SEED, study.RESAMPLING_SEED) == (20261641, 20261642)
        assert study.STUDY.publication_policy == "gated"
        assert (study.PRIMARY_N, study.PRIMARY_SCENARIO) == (4_000, "SW")
        assert study.PRIMARY_REPLICATES >= 1_000
        assert study.POINT_SIZES == (1_000, 4_000, 16_000)
        assert study.LONGITUDINAL_N == 4_000
        assert law.WEIGHT_COEFFICIENTS == (0.4, 0.5, 0.15)
        assert (law.G_BOUNDS, law.MAX_ITER, law.TOL, law.N_FOLDS) == ((0.01, 0.99), 100, 1e-10, 5)

    def test_the_seeds_are_the_study_s_own(self) -> None:
        seeds = {
            seed
            for record in registered()
            if record.slug != study.SLUG
            for seed in (record.seed, record.resampling_seed)
        }
        assert not {study.SEED, study.RESAMPLING_SEED, pilot.PILOT_SEED} & seeds
        assert law.INFLUENCE_SEED not in seeds

    def test_declared_and_not_registered(self) -> None:
        assert study.SLUG not in {record.slug for record in registered()}
        from tests.unit.test_method_evidence import UNCOMMITTED_MODULES

        assert properties.__name__ in UNCOMMITTED_MODULES

    def test_the_record_declares_the_cells_the_module_builds(self) -> None:
        built: dict[str, set[str]] = {}
        for cell in properties.declared_cells():
            built.setdefault(cell.property, set()).add(cell.cell)
        assert built == {family: set(cells) for family, cells in study.PROPERTY_CELLS.items()}

    def test_the_roles(self) -> None:
        roles = {(c.property, c.cell): c.role for c in properties.declared_cells()}
        for label in properties.CALIBRATION_LABELS:
            assert roles[("interval_calibration", f"{label}__correctly_specified")] == "positive"
            assert roles[("interval_calibration", f"{label}__noise_control")] == "control"
            assert roles[("sample_bias", label)] == "positive"
        for label in properties.POINT_LABELS:
            assert roles[("target_reading_control", label)] == "control"
            assert roles[("sample_misspecified", label)] == property_verdicts.DIAGNOSTIC_ROLE

    def test_each_size_has_its_own_stream_and_its_cells_share_it(self) -> None:
        keys = properties.payloads(budget=3)
        seeds: dict[str, list[int]] = {}
        for ((label, replicate, _),) in keys:
            seeds.setdefault(label, []).append(properties.sample_seed(label, replicate))
        assert set(seeds) == set(properties.CALIBRATION_LABELS)
        flat = [seed for values in seeds.values() for seed in values]
        assert len(set(flat)) == len(flat)
        # One key fills every cell of its label: the gated cell, the control and the
        # reporting cell of a size read one draw.
        for label in properties.POINT_LABELS:
            names = properties.cells_of(((label, 0, 1),))
            assert f"target_reading_control/{label}" in names
            assert f"sample_misspecified/{label}" in names
            assert f"sample_bias/{label}" in names

    def test_the_point_fits_read_the_declared_settings(self) -> None:
        assert {cell.fit for cell in properties.CELLS if cell.family == "sample_bias"} == {
            "sample_correct",
            "ltmle",
        }
        assert law.SETTINGS["sample_correct"] == law.Setting("sample", "correct")
        assert law.SETTINGS["target_main"] == law.Setting("target", "main")
        assert law.SETTINGS["sample_main"] == law.Setting("sample", "main")


# ------------------------------------------------------------------------------ law SW
class TestLawSW:
    def test_the_truth_reads_q_w_and_q_0_alike(self) -> None:
        """Plan review m6: ``Q_w = Q_0`` because the weight does not read ``Y``."""
        general, reduced = law.tilted_truth(law.sw_weight), law.q0_formula(law.sw_weight)
        for name in law.ESTIMANDS:
            assert general[name] == pytest.approx(reduced[name], abs=1e-14)
        assert law.truth()["ate"] == pytest.approx(0.330353, abs=5e-7)

    def test_a_weight_that_reads_y_separates_the_two_forms(self) -> None:
        general = law.tilted_truth(law.y_reading_weight)
        reduced = law.q0_formula(law.y_reading_weight)
        assert abs(general["ate"] - reduced["ate"]) > GUARD

    def test_the_tilted_truth_differs_from_the_unweighted_ate(self) -> None:
        assert law.unweighted_truth()["ate"] == pytest.approx(0.234120, abs=5e-7)
        assert law.truth()["ate"] - law.unweighted_truth()["ate"] > GUARD

    def test_the_truth_matches_a_monte_carlo_of_the_tilted_law(self) -> None:
        rng = np.random.default_rng(MC_SEED)
        w1 = rng.standard_normal(MC_DRAWS)
        w2 = (rng.random(MC_DRAWS) < 0.5).astype(float)
        a = (rng.random(MC_DRAWS) < law.g0(w2)).astype(float)
        w = law.weight(a, w1)
        blip = law.qbar0(1.0, w1, w2) - law.qbar0(0.0, w1, w2)
        terms = w * (blip - law.truth()["ate"])
        error = float(np.sqrt(np.mean(terms**2) / MC_DRAWS) / np.mean(w))
        assert float(np.sum(w * blip) / np.sum(w)) == pytest.approx(
            law.truth()["ate"], abs=5 * error
        )

    def test_the_tilt_is_a_main_terms_logistic_in_w1(self) -> None:
        """``m = w`` here, so ``P_v(A = 1 | W) = w(1, W) / sum_a w(a, W)``; and Bayes' rule."""
        at = law.grid()
        r = law.weight(1.0, at.w1) / (law.weight(0.0, at.w1) + law.weight(1.0, at.w1))
        np.testing.assert_allclose(law.tilt(at.w1), r, rtol=0, atol=1e-15)
        np.testing.assert_allclose(law.tilt(at.w1), expit(0.4 + 0.5 * at.w1), rtol=0, atol=1e-15)
        g = law.g0(at.w2)
        bayes = (
            g
            * law.weight(1.0, at.w1)
            / (g * law.weight(1.0, at.w1) + (1 - g) * law.weight(0.0, at.w1))
        )
        np.testing.assert_allclose(law.tilted_mechanism(at.w1, at.w2), bayes, rtol=0, atol=1e-14)

    def test_the_kish_ratio(self) -> None:
        assert law.kish_ratio() == pytest.approx(0.6540, abs=5e-4)
        for label in properties.POINT_LABELS:
            assert reading(block_of(label), "sample_correct")["kish"] == pytest.approx(
                law.kish_ratio(), abs=0.01
            )

    def test_the_bounds_never_bind(self) -> None:
        at = law.grid()
        g_w = law.tilted_mechanism(at.w1, at.w2)
        outside = (g_w < law.G_BOUNDS[0]) | (g_w > law.G_BOUNDS[1])
        assert at.mean(outside) < 1e-6

    def test_the_noise_arm_reads_the_exact_curve_sd(self) -> None:
        for label in properties.POINT_LABELS:
            assert properties.EFFICIENCY_SD[label] == pytest.approx(
                law.point_influence_sd(), abs=1e-6
            )


# ------------------------------------------------------------------------------ law LW
class TestLawLW:
    def test_the_longhand_truth_matches_the_weighted_g_formula(self) -> None:
        """The g-formula on ``P_w`` by a weighted Monte Carlo of ``L0`` and a nested ``L1``."""
        rng = np.random.default_rng(MC_SEED + 1)
        frame = law.draw_longitudinal(2_000_000, MC_SEED + 2)
        l0, w = frame["L0"].to_numpy(), frame["wt"].to_numpy()
        truth = law.longitudinal_truth()
        for name, arm in law.PLANS.items():
            a = float(arm)
            l1 = law.lw_l1_mean(l0, a) + rng.standard_normal(len(l0))
            value = float(np.sum(w * law.lw_q2(l0, a, l1, a)) / np.sum(w))
            assert value == pytest.approx(truth[f"ey_regimen[{name}]"], abs=0.01)
        assert truth[law.CONTRAST] == pytest.approx(1.769979, abs=5e-6)

    def test_the_tilt_moves_the_contrast(self) -> None:
        assert law.longitudinal_unweighted_truth()[law.CONTRAST] == pytest.approx(1.5)
        assert law.longitudinal_truth()[law.CONTRAST] - 1.5 > GUARD

    def test_the_first_node_is_tilted_and_the_second_is_not(self) -> None:
        """``P_w(A1 = 1 | L0)`` by Bayes; ``P_w(A2 | H2) = g2`` because ``w`` reads ``(L0, A1)``."""
        x = np.linspace(-4, 4, 81)
        g = law.lw_g1(x)
        bayes = (
            g
            * law.lw_weight(x, 1.0)
            / (g * law.lw_weight(x, 1.0) + (1 - g) * law.lw_weight(x, 0.0))
        )
        np.testing.assert_allclose(law.lw_tilted_first(x), bayes, rtol=0, atol=1e-14)
        assert np.max(np.abs(law.lw_tilted_first(x) - g)) > 0.1

    def test_the_kish_ratio_clears_the_floor(self) -> None:
        assert law.longitudinal_kish_ratio() >= KISH_FLOOR
        assert reading("lw_4000", "ltmle_sample")["kish"] >= KISH_FLOOR

    def test_the_noise_arm_reads_the_exact_curve_sd(self) -> None:
        value = law.longitudinal_influence_sd()
        assert properties.EFFICIENCY_SD[properties.LONGITUDINAL_LABEL] == pytest.approx(
            value, abs=5e-4
        )


# ------------------------------------------------------------------------- the control
class TestTheControl:
    def test_the_exact_bias_is_declared(self) -> None:
        assert law.control_limit()["bias"] == pytest.approx(properties.EXACT_CONTROL_BIAS, abs=1e-6)

    def test_the_main_terms_regression_is_wrong_and_the_correct_one_is_not(self) -> None:
        beta = law.main_terms_limit()
        at = law.grid()
        fitted = expit(beta[0] + beta[1] + beta[2] * at.w1 + beta[3] * at.w2)
        assert at.mean((fitted - law.qbar0(1.0, at.w1, at.w2)) ** 2) > 1e-3

    @pytest.mark.parametrize("label", properties.POINT_LABELS)
    def test_the_pilot_reads_the_exact_bias(self, label: str) -> None:
        row = reading(block_of(label), "target_main")
        error = float(row["sd"]) / np.sqrt(float(row["draws"]))
        assert row["bias"] == pytest.approx(properties.EXACT_CONTROL_BIAS, abs=5 * error + 0.003)

    @pytest.mark.parametrize("label", properties.POINT_LABELS)
    def test_the_control_is_powered(self, label: str) -> None:
        """The exact bias over the pilot's SD (its 99% upper limit) clears the margin."""
        row = reading(block_of(label), "target_main")
        draws = float(row["draws"])
        sd_upper = float(row["sd"]) * (1.0 + 2.576 / np.sqrt(2.0 * (draws - 1)))
        gap = abs(properties.EXACT_CONTROL_BIAS) / sd_upper
        power = sizing.bias_discriminated_probability(MARGINS, gap, properties.REPLICATES[label])
        assert power >= CONTROL_POWER

    def test_the_reporting_cell_removes_most_of_the_control_s_bias(self) -> None:
        for label in ("sw_n4000", "sw_n16000"):
            control = reading(block_of(label), "target_main")
            assert control["paired_difference_over_sd"] < -0.5


# --------------------------------------------------------------------------- the pilot
class TestThePilot:
    def test_the_pilot_is_the_shared_calibration_pilot_size(self) -> None:
        assert pilot.DRAWS == sizing.CALIBRATION_PILOT_DRAWS

    def test_the_pilot_ran_every_cell(self) -> None:
        for block, (_, n, fits) in pilot.BLOCKS.items():
            for fit in fits:
                row = reading(block, fit)
                assert int(row["draws"]) == pilot.DRAWS
                assert int(row["n"]) == n

    def test_the_pilot_truths_are_the_law_s(self) -> None:
        for label in properties.CALIBRATION_LABELS:
            row = reading(block_of(label), gated_fit(label))
            assert row["truth"] == pytest.approx(properties.truth_of(label), rel=1e-5)


# --------------------------------------------------------------------------- the sizes
def family_parts(replicates: dict[str, int], *, predictive: bool) -> dict[str, float]:
    """Each gated cell's pass probability at the declared replications."""
    parts: dict[str, float] = {}
    for label in properties.CALIBRATION_LABELS:
        r = replicates[label]
        row = reading(block_of(label), gated_fit(label))
        if predictive:
            parts[f"interval_calibration/{label}"] = pilot.predictive_calibration_pass(
                MARGINS, r, row
            )
            parts[f"sample_bias/{label}"] = pilot.predictive_bias_pass(MARGINS, r, row)
        else:
            parts[f"interval_calibration/{label}"] = pilot.calibration_cell_pass(
                MARGINS, r, row["se_ratio"], row["coverage"], row["ratio_spread"]
            )
            parts[f"sample_bias/{label}"] = float(
                sizing.bias_equivalent_probability(MARGINS, float(row["bias_over_sd"]), r)
            )
        for ratio in (properties.SHRUNKEN_SE_FACTOR, 1.0 / np.sqrt(2.0)):
            parts[f"derived/{label}/{ratio:.3f}"] = sizing.calibration_control_probability(
                MARGINS, r, ratio
            )
    for label in properties.POINT_LABELS:
        row = reading(block_of(label), "target_main")
        r = replicates[label]
        parts[f"target_reading_control/{label}"] = (
            pilot.predictive_control_pass(MARGINS, r, row)
            if predictive
            else sizing.bias_discriminated_probability(MARGINS, float(row["bias_over_sd"]), r)
        )
    return parts


def family(replicates: dict[str, int], *, predictive: bool) -> float:
    return float(np.prod(list(family_parts(replicates, predictive=predictive).values())))


class TestTheSizes:
    def test_the_replications_are_the_rule_s(self) -> None:
        """The smallest common multiple of 400 with predictive family pass >= 0.95."""

        def uniform(r: int) -> dict[str, int]:
            return dict.fromkeys(properties.CALIBRATION_LABELS, r)

        rule = next(
            r for r in range(400, 20_001, 400) if family(uniform(r), predictive=True) >= FAMILY_PASS
        )
        assert uniform(rule) == properties.REPLICATES

    def test_the_declared_pass_probabilities(self) -> None:
        plug_in = family(properties.REPLICATES, predictive=False)
        predictive = family(properties.REPLICATES, predictive=True)
        assert predictive >= FAMILY_PASS
        assert plug_in >= FAMILY_PASS
        assert (round(plug_in, 3), round(predictive, 3)) == DECLARED_FAMILY_PASS

    def test_each_calibration_cell_reads_above_ninety_percent(self) -> None:
        parts = family_parts(properties.REPLICATES, predictive=False)
        for label in properties.CALIBRATION_LABELS:
            assert parts[f"interval_calibration/{label}"] >= 0.90, label

    def test_the_derived_controls_fail_with_certainty(self) -> None:
        for label in properties.CALIBRATION_LABELS:
            for ratio in (properties.SHRUNKEN_SE_FACTOR, 1.0 / np.sqrt(2.0)):
                probability = sizing.calibration_control_probability(
                    MARGINS, properties.REPLICATES[label], ratio
                )
                assert probability > DERIVED_POWER

    def test_the_spread_rule_reduces_to_the_shared_one(self) -> None:
        """With normal estimates and an exact SE the SE half is the shared approximation's."""
        for r in (800, 2_400):
            shared = float(sizing.calibration_pass(MARGINS, r, np.array([1.0]), coverage=0.95)[0])
            ours = pilot.calibration_cell_pass(MARGINS, r, 1.0, 0.95, 1.0 / np.sqrt(2.0))
            assert ours == pytest.approx(shared, abs=0.01)

    def test_the_primary_passes(self) -> None:
        """The primary's three verdicts on the ATE at its size, from the gated n = 4,000 cell."""
        row = reading("sw_4000", "sample_correct")
        r = study.PRIMARY_REPLICATES
        bias = float(sizing.bias_equivalent_probability(MARGINS, float(row["bias_over_sd"]), r))
        coverage = sizing.coverage_floor_pass_probability(MARGINS, r, float(row["coverage"]))
        assert bias * coverage > 0.99
        assert MARGINS.se_ratio_sanity[0] < row["se_ratio"] < MARGINS.se_ratio_sanity[1]

    def test_the_budget(self) -> None:
        seconds = {
            (label, fit if fit != "ltmle_sample" else "ltmle"): float(
                reading(block_of(label), fit)["seconds"]
            )
            for label in properties.CALIBRATION_LABELS
            for fit in pilot.BLOCKS[block_of(label)][2]
        }
        assert properties.cpu_hours(seconds) == pytest.approx(DECLARED_CPU_HOURS, abs=0.05)


#: The plug-in and predictive family-wise pass probabilities at the declared replications.
DECLARED_FAMILY_PASS = (0.991, 0.962)
#: The fit CPU-hours of the declared run at the pilot's mean seconds per fit.
DECLARED_CPU_HOURS = 0.43


# ---------------------------------------------------------------------------- the smoke
def test_the_smoke_gate_passes() -> None:
    findings, seconds = properties.smoke_findings(draws=1)
    assert findings == []
    assert set(seconds) == {
        (label, fit) for label in properties.CALIBRATION_LABELS for fit in properties.fits_of(label)
    }


# ------------------------------------------------------------------ the red-cell reading
class TestTheRedCellReading:
    def test_the_gated_cells_are_every_cell_that_is_not_reported(self) -> None:
        declared = {
            f"{family}/{cell}"
            for family, cells in study.PROPERTY_CELLS.items()
            if family not in properties.REPORTED_FAMILIES
            for cell in cells
        }
        assert set(properties.gated_cells()) == declared
        assert len(declared) == 4 * 3 + 4 + 3

    def test_no_gated_cell_reads_an_inadmissible_red(self) -> None:
        """The declared red-cell reading, applied mechanically.

        Empty until the run commits its tables; at registration a red control, or a red
        positive without its finite-sample owner, fails here.
        """
        from tests.studies.mtp_study_tools import gating_findings

        findings = gating_findings(
            study.STUDY, properties.gated_cells(), truth_implementations=("cleverly",)
        )
        assert findings == []
