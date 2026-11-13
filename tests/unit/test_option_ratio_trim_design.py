"""The design of ``option-ratio-trim`` (Fix-8 plan sections 5.0 and 5.6), recomputed.

Every declared number of :mod:`tests.studies.canonical_option_ratio_trim` and
:mod:`tests.studies.option_ratio_trim_properties` is checked here: the exact truth, the law's
trim quantiles, the exact-law limit of every fitted cell, the replication count and its pass
probabilities, the pre-read of the committed pilot (``tests/studies/option_ratio_trim_pilot.csv``),
and the declared red-cell reading.

The bias cells read the standardized bias ``mean / SD`` of the estimates.  The trimmed fit's
sampling SD is not its plug-in curve SD (the plug-in SE ratio of a trimmed fit is far below 1),
so a control's limit in SD units is its exact bias over the pilot SD of its estimates.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pytest

from tests.studies import _option_trim_law as law
from tests.studies import canonical_option_ratio_trim as study
from tests.studies import option_ratio_trim_pilot as pilot
from tests.studies import option_ratio_trim_properties as properties
from tests.studies import repeated_sizing as sizing
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.registry import Margins, registered

MARGINS = Margins()
#: Plan 5.0: a bias positive whose pilot bias exceeds this raises its replications.
PILOT_BIAS_CEILING = 0.12
#: Plan 5.0: a control's discrimination power at its limit.
CONTROL_POWER = 0.9999
#: The joint pass probability the bias positives need at their pilot biases.
FAMILY_PASS = 0.95
#: Plan 5.6's replication count, which the rule may only raise.
PLANNED_REPLICATES = 1_600


@functools.cache
def limit(setting: str, nodes: int = law.QUADRATURE_NODES) -> law.ExactLimit:
    return law.exact_limit(setting, nodes)


def in_sample(setting: str) -> str:
    """The in-sample setting whose limit a cross-fitted one shares."""
    return setting.replace("_stacked_", "_in_sample_")


@pytest.fixture(scope="module")
def summary() -> pd.DataFrame:
    return pd.read_csv(pilot.SUMMARY)


def reading(summary: pd.DataFrame, setting: str) -> pd.Series:
    rows = summary.loc[summary["setting"] == setting]
    assert len(rows) == 1, setting
    return rows.iloc[0]


def cell_settings() -> dict[tuple[str, str], str]:
    return {
        (family, cell): setting
        for setting, outputs in properties.FITS.items()
        for family, cell in outputs
        if (family, cell) != properties.SHARE_CELL
    }


def positives() -> list[str]:
    return [
        setting
        for (family, cell), setting in cell_settings().items()
        if family in properties.BIAS_FAMILIES and properties._role(family, cell) == "positive"
    ]


def controls() -> list[str]:
    return [
        setting
        for (family, cell), setting in cell_settings().items()
        if family in properties.BIAS_FAMILIES and properties._role(family, cell) == "control"
    ]


def control_bias_over_sd(summary: pd.DataFrame, setting: str) -> float:
    return limit(setting).bias() / float(reading(summary, setting)["sd"])


#: The one registered study that shares this record's seed.  Parts D and E were declared in
#: parallel and both took 20267641/20267642, and the committed runs drew from it.  Their primary
#: replicates read the same integer seeds on different laws (point ``LDA``, longitudinal
#: ``LE``), so the samples are dependent across the two studies but not identical, and no
#: claim of either study reads the other.  Every property stream hashes a label of its own
#: (``stream_seed``), so the property draws do not share seeds.  Changing the seed would need
#: both studies rerun.
SHARED_SEED_STUDY = "option-treatment-adjustment"


class TestTheDeclaration:
    def test_the_constants(self) -> None:
        assert study.SLUG == "option-ratio-trim"
        assert (study.SEED, study.RESAMPLING_SEED) == (20267641, 20267642)
        assert study.STUDY.publication_policy == "reporting"
        assert study.STUDY.reference is None
        assert study.STUDY.point_only_scenarios == frozenset({"LE"})
        assert (study.PRIMARY_N, study.PRIMARY_REPLICATES) == (2_000, 1_600)
        assert study.PRIMARY_SETTING == "ltmle_in_sample_right_trim95"
        assert (law.TRIM, law.REPORTED_TRIM, law.N_FOLDS) == (0.95, 0.99, 5)
        assert (law.G_BOUNDS, law.MAX_ITER, law.TOL) == ((1e-8, 1.0), 100, 1e-10)

    def test_the_seeds_are_the_study_s_own(self) -> None:
        seeds = {
            s
            for record in registered()
            if record.slug not in {study.SLUG, SHARED_SEED_STUDY}
            for s in (record.seed, record.resampling_seed)
        }
        assert not {study.SEED, study.RESAMPLING_SEED, pilot.PILOT_SEED} & seeds
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
        assert built == {family: set(cells) for family, cells in study.PROPERTY_CELLS.items()}

    def test_the_roles_and_settings(self) -> None:
        cells = {(c.property, c.cell): c for c in properties.declared_cells()}
        for name in ("ltmle_in_sample", "ltmle_stacked", "sdr_in_sample", "sdr_stacked"):
            declared = cells[("trim_point", name)]
            setting = law.SETTINGS[declared.fit_kwargs["setting"]]
            assert declared.role == "positive"
            assert (setting.outcome, setting.trim) == ("right", law.TRIM)
            assert setting.cross_fit == name.endswith("stacked")
            assert setting.estimator == name.split("_")[0].upper()
        for estimator in ("ltmle", "sdr"):
            trimmed = cells[("trim_necessity", f"{estimator}_trimmed")]
            untrimmed = cells[("trim_necessity", f"{estimator}_untrimmed")]
            assert (trimmed.role, untrimmed.role) == ("control", "positive")
            assert law.SETTINGS[trimmed.fit_kwargs["setting"]].trim == law.TRIM
            assert law.SETTINGS[untrimmed.fit_kwargs["setting"]].trim is None
            for pair in (trimmed, untrimmed):
                assert law.SETTINGS[pair.fit_kwargs["setting"]].outcome == "wrong"
        for key, declared in cells.items():
            if key[0] == "trim_reported":
                assert declared.role == property_verdicts.DIAGNOSTIC_ROLE

    def test_every_fit_reads_the_same_draws(self) -> None:
        seeds: dict[str, list[int]] = {}
        for ((_, setting, _, _, _, seed, _),) in properties.payloads(budget=2):
            seeds.setdefault(setting, []).append(seed)
        assert set(seeds) == set(law.SETTINGS) == set(properties.FITS)
        assert len({tuple(values) for values in seeds.values()}) == 1


class TestTheLaw:
    def test_the_truth_matches_monte_carlo(self) -> None:
        rng = np.random.default_rng(11)
        w = rng.uniform(size=4_000_000)
        l2 = rng.binomial(1, law.p_covariate(w, 1.0))
        assert np.mean(law.p_outcome(w, l2, 1.0, 1.0)) == pytest.approx(law.TRUTH, abs=5e-4)

    def test_the_draws_have_the_law_s_moments(self) -> None:
        frame = law.draw(400_000, 3)
        w, a1 = frame["W"], frame["A1"]
        assert (a1 - law.p_treatment_1(w)).mean() == pytest.approx(0.0, abs=3e-3)
        seen = frame["L2"].notna()
        residual = frame.loc[seen, "L2"] - law.p_covariate(w[seen], a1[seen])
        assert residual.mean() == pytest.approx(0.0, abs=3e-3)
        assert frame["C1"].isin([0.0, 1.0]).all()

    def test_the_support_is_weak(self) -> None:
        """Low-W units have ``g_1 g_2`` near 0.005."""
        product = law.p_treatment_1(0.0) * law.p_treatment_2(0.0, 0.0, 1.0)
        assert 0.004 < product < 0.01

    def test_the_trim_quantiles(self) -> None:
        assert law.population_threshold(law.TRIM) == pytest.approx(4.4506, abs=1e-4)
        assert law.population_threshold(law.REPORTED_TRIM) == pytest.approx(10.4896, abs=1e-4)
        assert law.trimmed_share(law.TRIM) == pytest.approx(0.15783, abs=1e-5)

    def test_the_quadrature_has_converged(self) -> None:
        for setting in ("ltmle_in_sample_wrong_trim95", "sdr_in_sample_wrong_trim95"):
            assert limit(setting, 60).estimate == pytest.approx(limit(setting).estimate, abs=1e-6)


class TestTheExactLimits:
    @pytest.mark.parametrize("key", sorted(properties.EXACT_BIAS))
    def test_the_declared_limits(self, key: tuple[str, str]) -> None:
        setting = in_sample(cell_settings()[key])
        assert limit(setting).bias() == pytest.approx(properties.EXACT_BIAS[key], abs=1e-6)

    @pytest.mark.parametrize("key", sorted(properties.EXACT_BIAS))
    def test_the_zero_limits_are_exact(self, key: tuple[str, str]) -> None:
        if properties.EXACT_BIAS[key] == 0.0:
            assert abs(limit(in_sample(cell_settings()[key])).bias()) < 1e-8

    def test_the_ltmle_limit_is_the_trim_remainder(self) -> None:
        frame, result = law.limit_fit("ltmle_in_sample_wrong_trim95")
        remainder = law.trim_remainder(frame, result, law.TRIM)
        assert remainder == pytest.approx(
            properties.EXACT_BIAS[("trim_necessity", "ltmle_trimmed")], abs=1e-6
        )


class TestTheSizes:
    def test_the_controls_reach_the_floor(self, summary: pd.DataFrame) -> None:
        for setting in controls():
            assert control_bias_over_sd(summary, setting) >= law.CONTROL_FLOOR_SD

    def test_the_reported_trim_would_leave_the_control_under_the_floor(
        self, summary: pd.DataFrame
    ) -> None:
        """Why 0.95: trimmed at 0.99 a W-blind control does not reach 0.45 SD."""
        values = [
            control_bias_over_sd(summary, f"{estimator}_in_sample_wrong_trim99")
            for estimator in ("ltmle", "sdr")
        ]
        assert min(values) < law.CONTROL_FLOOR_SD

    def test_the_replication_count(self, summary: pd.DataFrame) -> None:
        def adequate(replicates: int) -> bool:
            joint = float(
                np.prod(
                    [
                        sizing.bias_equivalent_probability(
                            MARGINS, float(reading(summary, setting)["bias_over_sd"]), replicates
                        )
                        for setting in positives()
                    ]
                )
            )
            powers = [
                sizing.bias_discriminated_probability(
                    MARGINS, control_bias_over_sd(summary, setting), replicates
                )
                for setting in controls()
            ]
            return joint >= FAMILY_PASS and min(powers) > CONTROL_POWER

        rule = next(r for r in range(400, 20_001, 400) if adequate(r))
        assert max(PLANNED_REPLICATES, rule) == properties.REPLICATES

    def test_the_primary_reads_a_gated_positive(self) -> None:
        assert ("trim_point", "ltmle_in_sample") in {
            key for key, setting in cell_settings().items() if setting == study.PRIMARY_SETTING
        }


class TestThePreRead:
    def test_the_pilot_ran_its_draws_from_draw_zero(self, summary: pd.DataFrame) -> None:
        assert set(summary["setting"]) == set(law.SETTINGS)
        for setting in law.SETTINGS:
            row = reading(summary, setting)
            assert (int(row["draws"]), int(row["first_draw"])) == (pilot.DRAWS, 0)
            assert int(row["last_draw"]) == pilot.DRAWS - 1
            assert int(row["n"]) == properties.N

    @pytest.mark.parametrize("setting", sorted(set(positives())))
    def test_each_bias_positive_reads_below_the_ceiling(
        self, summary: pd.DataFrame, setting: str
    ) -> None:
        assert abs(reading(summary, setting)["bias_over_sd"]) <= PILOT_BIAS_CEILING

    @pytest.mark.parametrize("setting", sorted(set(controls())))
    def test_each_control_reads_near_its_limit(self, summary: pd.DataFrame, setting: str) -> None:
        row = reading(summary, setting)
        error = float(row["sd"]) / np.sqrt(float(row["draws"]))
        assert row["bias"] == pytest.approx(limit(setting).bias(), abs=5 * error)

    def test_the_trimmed_share_reads_near_its_limit(self, summary: pd.DataFrame) -> None:
        row = reading(summary, "ltmle_in_sample_right_trim95")
        assert row["trimmed_share"] == pytest.approx(law.trimmed_share(law.TRIM), abs=0.01)

    def test_the_plug_in_se_understates_the_trimmed_spread(self, summary: pd.DataFrame) -> None:
        """The diagnostic the package keeps is not the SE; the reason the interval is withheld."""
        assert reading(summary, "ltmle_in_sample_right_trim95")["se_ratio"] < 0.8


#: The cells whose red verdict gates the study: the four point positives and the two
#: necessity pairs.
GATED = (
    *(f"trim_point/{cell}" for cell in study.PROPERTY_CELLS["trim_point"]),
    *(f"trim_necessity/{cell}" for cell in study.PROPERTY_CELLS["trim_necessity"]),
)


class TestTheRedCellReading:
    def test_the_gated_cells_are_every_cell_that_is_not_reported(self) -> None:
        declared = {
            f"{family}/{cell}"
            for family, cells in study.PROPERTY_CELLS.items()
            if family not in properties.REPORTED_FAMILIES
            for cell in cells
        }
        assert len(GATED) == 8
        assert set(GATED) == declared

    def test_no_gated_cell_reads_an_inadmissible_red(self) -> None:
        """The declared red-cell reading, applied mechanically.

        Empty until the run commits its tables; at registration a red control, or a red
        positive without its finite-sample owner, fails here.
        """
        from tests.studies.mtp_study_tools import gating_findings

        assert gating_findings(study.STUDY, GATED, truth_implementations=("cleverly",)) == []
