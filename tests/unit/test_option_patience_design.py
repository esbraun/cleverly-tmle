"""The design of ``option-patience`` (Fix-8 plan sections 5.0 and 5.5), recomputed.

Every declared number of :mod:`tests.studies.canonical_option_patience` and
:mod:`tests.studies.option_patience_properties` is checked here: the exact LC3 truth, the bound
on the whole-path limit bias that the declared ``g_bounds`` leave on LC3, the choice of
``p*``, the mixture sizing of the late-confounder control at the pilot's 99% lower limits, the
pass probabilities of the bias, share and paired cells at the declared replications, the
pre-read of the committed pilot (``tests/studies/option_patience_pilot.csv``), and the declared
red-cell reading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from scipy.stats import beta, binom, norm

from tests.studies import _option_patience_laws as law
from tests.studies import canonical_option_patience as study
from tests.studies import option_patience_pilot as pilot
from tests.studies import option_patience_properties as properties
from tests.studies import repeated_sizing as sizing
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.registry import Margins, registered

MARGINS = Margins()
#: Plan 5.0: a bias positive whose pilot bias exceeds this raises its replications.
PILOT_BIAS_CEILING = 0.12
#: Plan 5.0: a control's limit floor, in SDs of the cell, and its discrimination power.
CONTROL_FLOOR = 0.45
CONTROL_POWER = 0.9999
#: The pass probability every gated positive must reach at its pilot reading.
POSITIVE_PASS = 0.99
#: Plan 5.5: ``p*`` is the smallest patience in this list whose greedy stop share reaches 0.5.
PATIENCE_CHOICES = (1, 2)
STOP_SHARE_FOR_P_STAR = 0.5


@pytest.fixture(scope="module")
def summary() -> pd.DataFrame:
    return pd.read_csv(pilot.SUMMARY)


def reading(summary: pd.DataFrame, cell: str) -> pd.Series:
    rows = summary.loc[summary["cell"] == cell]
    assert len(rows) == 1, cell
    return rows.iloc[0]


def tail() -> float:
    return (1.0 - MARGINS.confidence_level) / 2.0


def share_lower(share: float, draws: int) -> float:
    """The two-sided 99% Clopper-Pearson lower limit of a pilot share."""
    hits = round(share * draws)
    return float(beta.ppf(tail(), hits, draws - hits + 1)) if hits else 0.0


def sd_lower(sd: float, draws: int) -> float:
    """The 99% lower limit of an SD from ``draws`` draws (chi-square)."""
    from scipy.stats import chi2

    return float(sd * np.sqrt((draws - 1) / chi2.ppf(1.0 - tail(), draws - 1)))


def floor_pass_probability(replicates: int, rate: float, floor: float) -> float:
    """P(the exact interval's lower limit is at least ``floor``) at a true share ``rate``."""
    counts = np.arange(1, replicates + 1)
    passing = counts[beta.ppf(tail(), counts, replicates - counts + 1) >= floor]
    return float(binom.sf(int(passing[0]) - 1, replicates, rate)) if passing.size else 0.0


def mixture_bias(p: float, b: float) -> float:
    """The standardized bias of a mixture: share ``p`` at bias ``b``, the rest at 0, unit SDs."""
    return p * b / np.sqrt(1.0 + b**2 * p * (1.0 - p))


def paired_pass_probability(mean: float, sd: float, replicates: int) -> float:
    """P(the 99% Student interval of a mean paired difference lies inside +/- the margin).

    ``mean`` and ``sd`` are the paired differences' mean and SD in whole-path SDs.
    """
    z = float(norm.ppf(1.0 - tail()))
    half = z * sd / np.sqrt(replicates)
    se = sd / np.sqrt(replicates)
    margin = properties.PAIRED_MARGIN
    if margin <= half:
        return 0.0
    return float(norm.cdf((margin - half - mean) / se) - norm.cdf((-margin + half - mean) / se))


class TestTheDeclaration:
    def test_the_constants(self) -> None:
        assert study.SLUG == "option-patience"
        assert (study.SEED, study.RESAMPLING_SEED) == (20267541, 20267542)
        assert study.STUDY.publication_policy == "reporting"
        assert (study.PRIMARY_N, study.PRIMARY_REPLICATES, study.REPLICATES) == (
            2_000,
            1_600,
            1_600,
        )
        assert study.WITNESS_SHARE == 0.3
        assert study.STUDY.point_only_scenarios == frozenset({study.PRIMARY_SCENARIO})
        assert (law.LC2_PATIENCE, law.LC3_PATIENCE) == (2, 2)
        assert law.LC3_ORDERING == ("Z1", "Z2", "Z3", "W1")
        assert law.LC1_G_BOUNDS == law.LC2_G_BOUNDS == (0.025, 0.975)
        assert law.LC3_G_BOUNDS == (0.01, 0.99)
        assert (law.SELECTION_FOLDS, law.SELECTION_INNER_FOLDS) == (5, 2)
        assert (law.MAX_ITER, law.TOL) == (100, 1e-10)
        assert properties.PAIRED_MARGIN == MARGINS.standardized_bias == 0.25

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
        assert built == {family: set(cells) for family, cells in study.PROPERTY_CELLS.items()}

    def test_the_roles(self) -> None:
        roles = {(c.property, c.cell): c.role for c in properties.declared_cells()}
        assert roles.pop(("ordered_late_confounder", "lc3_ordered")) == "control"
        for (family, _), role in roles.items():
            expected = (
                property_verdicts.DIAGNOSTIC_ROLE
                if family in properties.REPORTED_FAMILIES
                else "positive"
            )
            assert role == expected

    def test_each_cell_reads_its_declared_fit(self) -> None:
        cells = {cell.key: cell for cell in properties.CELLS}
        assert law.FITS[cells["patience_bias/lc1_greedy"].fit].patience == law.LC1_PATIENCE
        assert law.FITS[cells["patience_bias/lc1_greedy"].fit].strategy == "greedy"
        control = law.FITS[cells["ordered_late_confounder/lc3_ordered"].fit]
        assert (control.strategy, control.patience) == ("ordered", 2)
        assert control.options["ordering"] == law.LC3_ORDERING
        whole = law.FITS[cells["ordered_whole_path/lc3_ordered"].fit]
        assert (whole.strategy, whole.patience) == ("ordered", None)
        paired = cells["greedy_late_confounder/lc3_greedy"]
        assert law.FITS[paired.fit].patience == 2
        assert law.FITS[str(paired.whole)].patience is None
        for cell in properties.CELLS:
            if cell.whole is not None:
                assert law.FITS[cell.whole].stream == law.FITS[cell.fit].stream
                assert law.FITS[cell.whole].strategy == law.FITS[cell.fit].strategy
                assert law.FITS[cell.whole].patience is None

    def test_one_key_fits_every_cell_of_its_stream(self) -> None:
        keys = properties.payloads(budget=2)
        assert len(keys) == 2 * len(law.STREAMS)
        covered = {name for key in keys for name in properties.cells_of(key)}
        assert covered == {cell.key for cell in properties.CELLS}


class TestTheLaws:
    def test_the_lc3_truth(self) -> None:
        exact = law.lc3_truth()
        assert exact["ate"] == pytest.approx(0.102948, abs=5e-7)
        assert law.lc3_truth(120)["ate"] == pytest.approx(exact["ate"], abs=1e-12)
        w1 = np.random.default_rng(7).standard_normal(2_000_000)
        monte = np.mean(law.lc3_qbar(1.0, w1) - law.lc3_qbar(0.0, w1))
        assert monte == pytest.approx(exact["ate"], abs=5e-4)

    def test_the_lc3_draws_follow_the_law(self) -> None:
        frame = law.lc3_draw(400_000, 3)
        index = law.LC3_INSTRUMENT * frame[["Z1", "Z2", "Z3"]].sum(axis=1) + 0.6 * frame["W1"]
        assert (frame["A"] - expit(index)).mean() == pytest.approx(0.0, abs=3e-3)
        residual = frame["Y"] - law.lc3_qbar(frame["A"].to_numpy(dtype=float), frame["W1"])
        assert residual.mean() == pytest.approx(0.0, abs=3e-3)
        assert index.std() == pytest.approx(law.lc3_index_sd(), rel=5e-3)

    def test_the_sibling_truths_are_the_sibling_studies(self) -> None:
        from tests.studies import canonical_ctmle_selector as selector
        from tests.studies import canonical_ctmle_sequences as sequences

        _, truth = selector.draw_from_seed("binary_greedy", 50, 1)
        assert law.truth("lc1")["ate"] == truth["ate"]
        for stream, scenario in law.LC2_SCENARIOS.items():
            _, truth = sequences.draw_from_seed(scenario, 50, 1)
            assert law.truth(stream)["ate"] == truth["ate"]

    def test_the_learners_split_is_balanced_and_seeded(self) -> None:
        folds = law.selection_folds(2_000, 11)
        assert np.bincount(folds)[1:].tolist() == [400] * 5
        assert np.array_equal(folds, law.selection_folds(2_000, 11))
        assert not np.array_equal(folds, law.selection_folds(2_000, 12))


class TestTheWholePathLimit:
    """Plan 5.0's Part C note: the LC3 whole-path limit bias, with ``g_bounds`` declared."""

    def test_the_bound_matches_monte_carlo(self) -> None:
        rng = np.random.default_rng(5)
        index = law.lc3_index_sd() * rng.standard_normal(4_000_000)
        g = expit(index)
        truncated = np.clip(g, *law.LC3_G_BOUNDS)
        monte = np.mean(np.abs(1 - g / truncated) + np.abs(1 - (1 - g) / (1 - truncated)))
        assert law.lc3_truncation_bound() == pytest.approx(monte, rel=0.05)

    def test_untruncated_the_limit_is_zero(self) -> None:
        assert law.lc3_truncation_bound((1e-12, 1.0 - 1e-12)) < 1e-12

    def test_the_bound_is_below_the_pilot_ceiling(self, summary: pd.DataFrame) -> None:
        """The limit bound, plus the finite-sample W1-exclusion term, sizes the cell.

        The limit bound holds as n grows.  At n = 2,000 the whole path still drops ``W1`` on a
        share of draws, so the sized bias adds that share times its conditional bias, both at
        their 99% upper limits.  That sum sits just above the 0.12 ceiling, which gates the pilot
        reading (0.023 SD, exclusions included) in ``test_each_bias_positive_reads_below_the_ceiling``;
        here the cell must still pass at the adverse limits.
        """
        row = reading(summary, "lc3_ordered_whole")
        draws = int(row["draws"])
        unit = sd_lower(float(row["sd"]), draws)
        limit = law.lc3_truncation_bound() / unit
        assert limit < 0.03
        excluded = round(float(row["exclude_share"]) * draws)
        share = float(beta.ppf(1.0 - tail(), excluded + 1, draws - excluded))
        half = norm.ppf(1.0 - tail()) * float(row["conditional_bias_sd"]) / np.sqrt(excluded)
        conditional = (abs(float(row["conditional_bias"])) + half) / float(row["sd"])
        sized = limit + share * conditional
        assert sized == pytest.approx(0.124, abs=0.002)
        assert sizing.bias_equivalent_probability(MARGINS, sized, study.REPLICATES) > POSITIVE_PASS

    def test_the_default_bounds_would_leave_more(self, summary: pd.DataFrame) -> None:
        """The sample-size default at n = 2,000 truncates more, which is why LC3 declares one."""
        auto = 5.0 / (np.sqrt(law.N) * np.log(law.N))
        row = reading(summary, "lc3_ordered_whole")
        unit = sd_lower(float(row["sd"]), int(row["draws"]))
        assert law.lc3_truncation_bound((auto, 1.0 - auto)) / unit > 0.05
        assert law.lc3_truncated_share() < 0.003


class TestThePreRead:
    def test_the_pilot_ran_its_draws_from_draw_zero(self, summary: pd.DataFrame) -> None:
        assert set(summary["cell"]) == {n for names in pilot.STREAM_FITS.values() for n in names}
        for _, row in summary.iterrows():
            assert (int(row["draws"]), int(row["first_draw"])) == (pilot.DRAWS, 0)
            assert int(row["last_draw"]) == pilot.DRAWS - 1

    def test_p_star_is_the_smallest_patience_with_half_the_draws_stopped(
        self, summary: pd.DataFrame
    ) -> None:
        shares = {
            p: float(reading(summary, f"lc1_greedy_patience_{p}")["stop_share"])
            for p in PATIENCE_CHOICES
        }
        chosen = next(p for p in PATIENCE_CHOICES if shares[p] >= STOP_SHARE_FOR_P_STAR)
        assert chosen == law.LC1_PATIENCE

    @pytest.mark.parametrize(
        "cell", [f"lc1_greedy_patience_{law.LC1_PATIENCE}", "lc3_ordered_whole"]
    )
    def test_each_bias_positive_reads_below_the_ceiling(
        self, summary: pd.DataFrame, cell: str
    ) -> None:
        row = reading(summary, cell)
        assert abs(row["bias_over_sd"]) <= PILOT_BIAS_CEILING
        probability = sizing.bias_equivalent_probability(
            MARGINS, float(row["bias_over_sd"]), study.REPLICATES
        )
        assert probability > POSITIVE_PASS

    def test_the_stop_witness_clears_its_floor(self, summary: pd.DataFrame) -> None:
        row = reading(summary, f"lc1_greedy_patience_{law.LC1_PATIENCE}")
        lower = share_lower(float(row["stop_share"]), int(row["draws"]))
        assert lower >= study.WITNESS_SHARE
        probability = floor_pass_probability(study.REPLICATES, lower, study.WITNESS_SHARE)
        assert probability > CONTROL_POWER

    def test_the_paired_positive_passes(self, summary: pd.DataFrame) -> None:
        row = reading(summary, "lc3_greedy_patience")
        mean = float(row["paired_mean_over_whole_sd"])
        sd = float(row["paired_sd_over_whole_sd"])
        assert abs(mean) <= PILOT_BIAS_CEILING
        assert paired_pass_probability(mean, sd, study.REPLICATES) > POSITIVE_PASS
        # The same reading at a three-fold paired SD, so a pilot SD read low does not carry it.
        assert paired_pass_probability(mean, 3.0 * sd, study.REPLICATES) > POSITIVE_PASS
        assert float(row["identical_share"]) > 0.9


class TestTheLateConfounderControl:
    """Plan 5.5's mixture sizing at the pilot's 99% lower limits (review findings 6 and 4)."""

    def lower_limits(self, summary: pd.DataFrame) -> tuple[float, float]:
        row = reading(summary, "lc3_ordered_patience")
        whole = reading(summary, "lc3_ordered_whole")
        draws = int(row["draws"])
        p = share_lower(float(row["exclude_share"]), draws)
        stopped = round(float(row["exclude_share"]) * draws)
        z = float(norm.ppf(1.0 - tail()))
        conditional = float(row["conditional_bias"]) - z * float(row["conditional_bias_sd"]) / (
            np.sqrt(stopped)
        )
        return p, conditional / float(whole["sd"])

    def test_the_stop_witness_clears_its_floor(self, summary: pd.DataFrame) -> None:
        p, _ = self.lower_limits(summary)
        assert p >= study.WITNESS_SHARE
        assert floor_pass_probability(study.REPLICATES, p, study.WITNESS_SHARE) > CONTROL_POWER

    def test_the_mixture_limit_reaches_the_floor(self, summary: pd.DataFrame) -> None:
        p, b = self.lower_limits(summary)
        limit = mixture_bias(p, b)
        assert limit >= CONTROL_FLOOR
        assert limit <= np.sqrt(p / (1.0 - p))
        assert sizing.bias_discriminated_probability(MARGINS, limit, study.REPLICATES) > (
            CONTROL_POWER
        )

    def test_the_limit_is_insensitive_to_the_conditional_bias(self, summary: pd.DataFrame) -> None:
        p, _ = self.lower_limits(summary)
        assert mixture_bias(p, 1.0) >= CONTROL_FLOOR

    def test_the_formula_reads_the_pilot_cell(self, summary: pd.DataFrame) -> None:
        """The mixture formula at the pilot's point readings reproduces the cell's reading."""
        row = reading(summary, "lc3_ordered_patience")
        whole = reading(summary, "lc3_ordered_whole")
        p = float(row["exclude_share"])
        b = float(row["conditional_bias"]) / float(whole["sd"])
        assert mixture_bias(p, b) == pytest.approx(float(row["bias_over_sd"]), abs=0.15)
        assert float(row["bias_over_sd"]) >= CONTROL_FLOOR


#: The cells whose red verdict gates the study.
GATED = properties.gated_cells()


class TestTheRedCellReading:
    def test_the_gated_cells_are_every_cell_that_is_not_reported(self) -> None:
        declared = {
            f"{family}/{cell}"
            for family, cells in study.PROPERTY_CELLS.items()
            if family not in properties.REPORTED_FAMILIES
            for cell in cells
        }
        assert len(GATED) == 6
        assert set(GATED) == declared

    def test_no_gated_cell_reads_an_inadmissible_red(self) -> None:
        """The declared red-cell reading, applied mechanically.

        Empty until the run commits its tables; at registration a red control, or a red
        positive without its finite-sample owner, fails here.
        """
        from tests.studies.mtp_study_tools import gating_findings

        # The primary is point-only (F18), so its bias leg is the only truth leg that gates.
        assert gating_findings(study.STUDY, GATED, truth_implementations=("cleverly",)) == []
