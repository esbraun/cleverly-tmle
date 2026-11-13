"""The design of ``option-markov-order`` (Fix-8 plan sections 5.0 and 5.7), recomputed.

Every declared number of :mod:`tests.studies.canonical_option_markov_order` and
:mod:`tests.studies.option_markov_order_properties` is checked here: the exact truths, the
exact-law limits of the bias and reported cells, the efficiency bounds that size the noise
arms, the replication counts and their pass probabilities, the pre-read of the committed pilot
(``tests/studies/option_markov_order_pilot.csv``), and the declared red-cell reading.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from tests.studies import _option_markov_law as law
from tests.studies import canonical_option_markov_order as study
from tests.studies import option_markov_order_pilot as pilot
from tests.studies import option_markov_order_properties as properties
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
#: Plan 5.0's product floor of the three calibration cells.
PRODUCT_FLOOR = 2_800
N = 2_000

#: Each limit cell's setting and estimand.
LIMIT_SETTINGS = {
    ("order_necessity", "ate"): ("lf1_order0", law.ATE),
    ("order_double_robustness", "ey_never"): ("lf2_order0", law.EY_NEVER),
    (
        "order_double_robustness",
        "ey_never_blind_mechanisms",
    ): ("lf2_order0_blind", law.EY_NEVER),
    ("markov_reported", "full_history_ey"): ("lf1_full", law.EY_ALWAYS),
    ("markov_reported", "full_history_ate"): ("lf1_full", law.ATE),
    ("markov_reported", "order0_ey"): ("lf1_order0", law.EY_ALWAYS),
    ("markov_reported", "lf2_order0_ey"): ("lf2_order0", law.EY_ALWAYS),
    ("markov_reported", "lf2_order0_ate"): ("lf2_order0", law.ATE),
}
#: Each K label's setting and estimand.
CALIBRATION_SETTINGS = {
    "order1_ey": ("lf1_order1", law.EY_ALWAYS),
    "order1_ate": ("lf1_order1", law.ATE),
    "order2_ate": ("lf1_order2", law.ATE),
}


@functools.cache
def limits(name: str, nodes: int = law.QUADRATURE_NODES) -> dict[str, law.ExactLimit]:
    return law.exact_limit(name, nodes)


def exact_bias(name: str, estimand: str, nodes: int = law.QUADRATURE_NODES) -> float:
    return limits(name, nodes)[estimand].estimate - law.truth(law.SETTINGS[name].law)[estimand]


def bias_over_sd(name: str, estimand: str, nodes: int = law.QUADRATURE_NODES) -> float:
    """The limit's bias over its plug-in curve SD at ``N``, not the sampling SD."""
    return exact_bias(name, estimand, nodes) / limits(name, nodes)[estimand].sd(N)


@functools.cache
def mutant_bias_over_sd(estimand: str) -> float:
    """The ``LF2`` order-0 limit under the shipped mutation: mechanism arms start at ``t-k``."""
    import importlib
    import inspect
    import textwrap

    from cleverly.longitudinal import LongitudinalData

    data_module = importlib.import_module("cleverly.longitudinal.data")
    source = textwrap.dedent(inspect.getsource(LongitudinalData.history_design))
    old = "max(1, self.window_start(time) - 1)"
    assert source.count(old) == 1
    namespace: dict[str, object] = {}
    exec(
        compile(source.replace(old, "self.window_start(time)"), "<m>", "exec"),
        vars(data_module),
        namespace,
    )
    original = LongitudinalData.history_design
    LongitudinalData.history_design = namespace["history_design"]  # type: ignore[method-assign, assignment]
    try:
        limit = law.exact_limit("lf2_order0")[estimand]
    finally:
        LongitudinalData.history_design = original  # type: ignore[method-assign]
    return (limit.estimate - law.truth("LF2")[estimand]) / limit.sd(N)


@pytest.fixture(scope="module")
def summary() -> pd.DataFrame:
    return pd.read_csv(pilot.SUMMARY)


def reading(summary: pd.DataFrame, cell: str) -> pd.Series:
    rows = summary.loc[summary["cell"] == cell]
    assert len(rows) == 1, cell
    return rows.iloc[0]


def pilot_cell(setting: str, estimand: str) -> str:
    return f"{setting}:{estimand}"


def regimen_monte_carlo(name: str, a: float, n: int, seed: int) -> float:
    """``E[Y^a]`` by simulating the covariates under the static regimen ``a``."""
    rng = np.random.default_rng(seed)
    w = rng.uniform(size=n)
    l_prev, ls = np.zeros(n), []
    for t in range(1, law.T + 1):
        a_prev = np.zeros(n) if t == 1 else np.full(n, a)
        l_prev = rng.binomial(1, law.p_covariate(w, l_prev, a_prev)).astype(float)
        ls.append(l_prev)
    return float(np.mean(law.p_outcome(name, w, ls[3], ls[2], a, a)))


class TestTheDeclaration:
    def test_the_constants(self) -> None:
        assert study.SLUG == "option-markov-order"
        assert (study.SEED, study.RESAMPLING_SEED) == (20267651, 20267652)
        assert study.STUDY.publication_policy == "reporting"
        assert study.STUDY.reference is None
        assert (study.PRIMARY_N, study.PRIMARY_REPLICATES) == (2_000, 2_800)
        assert study.PRIMARY_SETTING == "lf1_order1"
        assert properties.ROBUSTNESS_REPLICATES == 1_600
        assert (law.G_BOUNDS, law.MAX_ITER, law.TOL) == ((1e-8, 1.0), 100, 1e-10)
        assert {name: s.order for name, s in law.SETTINGS.items()} == {
            "lf1_order1": 1,
            "lf1_order2": 2,
            "lf1_full": None,
            "lf1_order0": 0,
            "lf2_order0": 0,
            "lf2_order0_blind": 0,
        }

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

    def test_the_roles_and_estimands(self) -> None:
        cells = {(c.property, c.cell): c for c in properties.declared_cells()}
        for label, (setting, estimand) in CALIBRATION_SETTINGS.items():
            cell = cells[("interval_calibration", f"{label}__correctly_specified")]
            assert (cell.role, cell.estimand, cell.fit_kwargs["setting"]) == (
                "positive",
                estimand,
                setting,
            )
        for key, (setting, estimand) in LIMIT_SETTINGS.items():
            assert (cells[key].estimand, cells[key].fit_kwargs["setting"]) == (estimand, setting)
        assert cells[("order_necessity", "ate")].role == "control"
        assert cells[("order_double_robustness", "ey_never")].role == "positive"
        assert cells[("order_double_robustness", "ey_never_blind_mechanisms")].role == "control"
        assert law.SETTINGS["lf2_order0_blind"].blind_mechanisms
        assert not law.SETTINGS["lf2_order0"].blind_mechanisms
        for key, cell in cells.items():
            if key[0] == "markov_reported":
                assert cell.role == property_verdicts.DIAGNOSTIC_ROLE

    def test_the_calibration_fits_share_their_draws(self) -> None:
        seeds: dict[str, list[int]] = {}
        for ((_, setting, _, _, _, seed, _),) in properties.payloads(budget=2):
            seeds.setdefault(setting, []).append(seed)
        assert seeds["lf1_order1"] == seeds["lf1_order2"] == seeds["lf1_full"]
        # The calibration stream and the three bias cells' own.
        assert len({tuple(values) for values in seeds.values()}) == 4

    def test_the_primary_reads_the_order_one_fit(self) -> None:
        assert law.SETTINGS[study.PRIMARY_SETTING] == law.SETTINGS["lf1_order1"]
        assert study.STUDY.scenarios == {"LF1": law.ESTIMANDS}


class TestTheLaws:
    @pytest.mark.parametrize("name", law.LAWS)
    def test_the_truths_match_monte_carlo(self, name: str) -> None:
        truth = law.truth(name)
        for a, estimand in ((1.0, law.EY_ALWAYS), (0.0, law.EY_NEVER)):
            simulated = regimen_monte_carlo(name, a, 2_000_000, 11)
            assert simulated == pytest.approx(truth[estimand], abs=1e-3)
        assert truth[law.ATE] == pytest.approx(truth[law.EY_ALWAYS] - truth[law.EY_NEVER])

    def test_the_lf2_outcome_is_a_probability_leaning_on_l3(self) -> None:
        c0, cw, c4, c3, ca4, ca3 = law.OUTCOME["LF2"]
        corners = [
            c0 + cw * w + c4 * l4 + c3 * l3 + ca4 * a4 + ca3 * a3
            for w in (0, 1) for l4 in (0, 1) for l3 in (0, 1) for a4 in (0, 1) for a3 in (0, 1)
        ]  # fmt: skip
        assert min(corners) > 0.0 and max(corners) < 1.0
        assert c3 > law.OUTCOME["LF1"][3]

    def test_the_draws_have_the_law_s_moments(self) -> None:
        frame = law.draw("LF1", 400_000, 3)
        rows = frame["L1"].notna() & frame["L2"].notna()
        w, l1, a1, l2 = (frame.loc[rows, c] for c in ("W", "L1", "A1", "L2"))
        assert (l2 - law.p_covariate(w, l1, a1)).mean() == pytest.approx(0.0, abs=3e-3)
        a2 = frame.loc[rows, "A2"]
        assert (a2 - law.p_treatment("LF1", w, l2, a1, l1)).mean() == pytest.approx(0.0, abs=3e-3)

    @pytest.mark.parametrize("name", ["lf1_order0", "lf1_full"])
    def test_the_quadrature_has_converged(self, name: str) -> None:
        for estimand in law.ESTIMANDS:
            fine, coarse = limits(name, 30)[estimand], limits(name)[estimand]
            assert fine.estimate == pytest.approx(coarse.estimate, abs=1e-8)
            assert fine.curve_sd == pytest.approx(coarse.curve_sd, abs=1e-8)

    def test_the_efficiency_bounds_size_the_noise_arms(self) -> None:
        assert set(properties.EFFICIENCY_SD) == set(properties.CALIBRATION_LABELS)
        for label, value in properties.EFFICIENCY_SD.items():
            setting, estimand = CALIBRATION_SETTINGS[label]
            assert value == pytest.approx(limits(setting)[estimand].curve_sd, abs=1e-6)
            # At the truth the window's curve is the full-history one.
            assert value == pytest.approx(limits("lf1_full")[estimand].curve_sd, abs=1e-6)

    def test_the_declared_law_supports_the_always_path_more_than_the_discarded_one(
        self,
    ) -> None:
        """The rev 2 law starved ``EY(always)``; the full-history fit missed coverage too."""

        def always(treatment: Sequence[float] | None) -> float:
            frame = law.draw("LF1", 400_000, 5, treatment=treatment)
            arms = frame[[f"A{t}" for t in range(1, law.T + 1)]]
            return float((arms == 1.0).all(axis=1).mean())

        assert always(None) > 2.0 * always(law.DISCARDED_TREATMENT)
        assert "0.919" in str(study.CONFIGURATION["discarded_law"])


class TestTheExactLimits:
    @pytest.mark.parametrize("key", sorted(properties.EXACT_LIMITS))
    def test_the_declared_limits(self, key: tuple[str, str]) -> None:
        setting, estimand = LIMIT_SETTINGS[key]
        assert bias_over_sd(setting, estimand) == pytest.approx(
            properties.EXACT_LIMITS[key], abs=1e-4
        )

    @pytest.mark.parametrize("label", sorted(CALIBRATION_SETTINGS))
    def test_every_calibration_positive_is_unbiased(self, label: str) -> None:
        assert abs(bias_over_sd(*CALIBRATION_SETTINGS[label])) < 1e-6

    def test_the_controls_reach_the_floor(self) -> None:
        assert bias_over_sd("lf1_order0", law.ATE) >= law.CONTROL_FLOOR_SD
        assert bias_over_sd("lf2_order0_blind", law.EY_NEVER) >= law.CONTROL_FLOOR_SD

    def test_the_ate_control_could_not_reach_the_floor(self) -> None:
        """Why the pair reads ``EY(never)``: its ATE control is under 0.45 plug-in SDs."""
        assert abs(bias_over_sd("lf2_order0_blind", law.ATE)) < law.CONTROL_FLOOR_SD

    def test_the_shipped_mutation_lands_beyond_the_margin(self) -> None:
        """A mechanism window without ``A_{t-1}`` would turn the positive red."""
        bias = mutant_bias_over_sd(law.EY_NEVER)
        assert abs(bias) >= law.CONTROL_FLOOR_SD
        power = sizing.bias_discriminated_probability(
            MARGINS, abs(bias), properties.ROBUSTNESS_REPLICATES
        )
        assert power > CONTROL_POWER


class TestTheSizes:
    def test_the_product_floor_of_three_calibration_cells(self) -> None:
        def product(replicates: int) -> float:
            return float(sizing.calibration_pass(MARGINS, replicates, np.array([1.0]))[0]) ** 3

        floor = next(r for r in range(400, 20_001, 400) if product(r) >= 0.95)
        assert floor == PRODUCT_FLOOR
        assert product(floor) == pytest.approx(0.977, abs=2e-3)

    def test_the_pooled_predictive_rule(self, summary: pd.DataFrame) -> None:
        names = {label: pilot_cell(*CALIBRATION_SETTINGS[label]) for label in CALIBRATION_SETTINGS}
        ratios = {
            f"{label}__correctly_specified": float(reading(summary, cell)["se_ratio"])
            for label, cell in names.items()
        }

        def correlations(labels: Sequence[str]) -> np.ndarray:
            return np.array(
                [
                    [float(reading(summary, names[a])[f"correlation[{names[b]}]"]) for b in labels]
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

    @pytest.mark.parametrize(
        ("setting", "estimand"),
        [("lf1_order0", law.ATE), ("lf2_order0_blind", law.EY_NEVER)],
    )
    def test_each_control_is_discriminated(self, setting: str, estimand: str) -> None:
        power = sizing.bias_discriminated_probability(
            MARGINS, bias_over_sd(setting, estimand), properties.ROBUSTNESS_REPLICATES
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

    def test_the_pilot_fits_as_the_property_phase_fits(self) -> None:
        for block, (name, n, settings) in pilot.BLOCKS.items():
            for setting in settings:
                assert law.SETTINGS[setting].law == name, block
                _, size, _, _ = properties.FITS[setting]
                assert size == n

    def test_the_pilot_ran_its_draws_from_draw_zero(self, summary: pd.DataFrame) -> None:
        for block, (_, _, settings) in pilot.BLOCKS.items():
            draws = pilot.CALIBRATION_DRAWS if block == "calibration" else pilot.DRAWS
            for setting in settings:
                estimands = (
                    law.ESTIMANDS if law.SETTINGS[setting].law == "LF2" else law.ESTIMANDS[::2]
                )
                for estimand in estimands:
                    row = reading(summary, pilot_cell(setting, estimand))
                    assert (int(row["draws"]), int(row["first_draw"])) == (draws, 0)
                    assert int(row["last_draw"]) == draws - 1

    @pytest.mark.parametrize("label", sorted(CALIBRATION_SETTINGS))
    def test_each_gated_calibration_cell_reads_above_the_pre_read_floor(
        self, summary: pd.DataFrame, label: str
    ) -> None:
        row = reading(summary, pilot_cell(*CALIBRATION_SETTINGS[label]))
        assert row["coverage"] >= PRE_READ_COVERAGE
        replicates = properties.CALIBRATION_REPLICATES
        assert sizing.calibration_pass(MARGINS, replicates, np.array([row["se_ratio"]]))[0] > 0.5

    def test_the_bias_positive_reads_below_the_ceiling(self, summary: pd.DataFrame) -> None:
        row = reading(summary, pilot_cell("lf2_order0", law.EY_NEVER))
        assert abs(row["bias_over_sd"]) <= PILOT_BIAS_CEILING
        probability = sizing.bias_equivalent_probability(
            MARGINS, float(row["bias_over_sd"]), properties.ROBUSTNESS_REPLICATES
        )
        assert probability > 0.99

    @pytest.mark.parametrize(
        ("setting", "estimand"),
        [("lf1_order0", law.ATE), ("lf2_order0_blind", law.EY_NEVER)],
    )
    def test_each_control_reads_near_its_limit(
        self, summary: pd.DataFrame, setting: str, estimand: str
    ) -> None:
        """In outcome units: the blind fit's plug-in SD is not its sampling SD."""
        row = reading(summary, pilot_cell(setting, estimand))
        error = float(row["sd"]) / np.sqrt(float(row["draws"]))
        assert row["bias"] == pytest.approx(exact_bias(setting, estimand), abs=5 * error)
        assert row["bias_over_sd"] >= bias_over_sd(setting, estimand) - 0.15

    def test_the_full_history_fit_reads_beside_the_window(self, summary: pd.DataFrame) -> None:
        """The diagnosis of a red K cell reads the full-history fit on the same draws."""
        for estimand in (law.EY_ALWAYS, law.ATE):
            row = reading(summary, pilot_cell("lf1_full", estimand))
            assert row["coverage"] >= PRE_READ_COVERAGE


#: The cells whose red verdict gates the study: the calibration positives, their derived
#: controls, the necessity control and the double-robustness pair.
GATED = (
    *(f"interval_calibration/{label}__correctly_specified" for label in study.CALIBRATION_LABELS),
    *(name for derived in properties.DERIVED_CELLS.values() for name in derived),
    "order_necessity/ate",
    "order_double_robustness/ey_never",
    "order_double_robustness/ey_never_blind_mechanisms",
)


class TestTheRedCellReading:
    def test_the_gated_cells_are_every_cell_that_is_not_reported(self) -> None:
        declared = {
            f"{family}/{cell}"
            for family, cells in study.PROPERTY_CELLS.items()
            if family not in properties.REPORTED_FAMILIES
            for cell in cells
        }
        assert len(GATED) == 12
        assert set(GATED) == declared

    def test_no_gated_cell_reads_an_inadmissible_red(self) -> None:
        """The declared red-cell reading, applied mechanically.

        Empty until the run commits its tables; at registration a red control, or a red
        positive without its finite-sample owner, fails here.
        """
        from tests.studies.mtp_study_tools import gating_findings

        assert gating_findings(study.STUDY, GATED, truth_implementations=("cleverly",)) == []
