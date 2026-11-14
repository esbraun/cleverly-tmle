"""The design of ``known-score-att`` (X35 plan section 5, rev 2.1), recomputed.

Every declared number of :mod:`tests.studies.known_score_att` and
:mod:`tests.studies.known_score_att_properties` is checked here: the exact truths and curve
SDs of Law S, Law S0 and three-arm Law B, the control's population bias, the replication
counts and the pass probability of each gated instrument, the control power, the designed
chance of a red cell, the pre-read of the committed pilot
(``tests/studies/known_score_att_pilot.csv``) and the declared red-cell reading.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from tests.studies import known_mechanism_law as law_b
from tests.studies import known_score_att as study
from tests.studies import known_score_att_law as law
from tests.studies import known_score_att_pilot as pilot
from tests.studies import known_score_att_properties as properties
from tests.studies import repeated_sizing as sizing
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.registry import Margins, registered

MARGINS = Margins()
Z = float(norm.ppf(0.5 + MARGINS.confidence_level / 2.0))
#: A gated calibration cell whose pilot coverage reads below this is diagnosed first.
PRE_READ_COVERAGE = 0.92
#: A bias positive whose pilot bias exceeds this raises its replications.
PILOT_BIAS_CEILING = 0.12
#: A control's discrimination power, and a positive's pass probability where its margin is
#: wide.
CERTAIN = 0.9999
#: The guide's floor on a calibration label's plug-in pass probability at its pilot ratio.
PLUG_IN_FLOOR = 0.90
#: The calibration instrument's pass floor: its share of the study-wide 0.95
#: (:data:`tests.studies.repeated_sizing.SIZED_SHARE`).
FAMILY_FLOOR = sizing.SIZED_SHARE
#: Each property family's floor: two families (accuracy, the root-n ladder) share the property
#: instrument's failure.
PROPERTY_FLOOR = sizing.family_share(2)
#: The at-target size of the six calibration positives (a product at SE ratio one).
AT_TARGET = 3_200
#: The designed chance of a red gated cell, declared in the X35 study declaration.
DESIGNED_RED = 0.025


@pytest.fixture(scope="module")
def summary() -> pd.DataFrame:
    return pd.read_csv(pilot.SUMMARY)


def reading(summary: pd.DataFrame, block: str, setting: str, estimand: str) -> pd.Series:
    rows = summary.loc[
        (summary["block"] == block)
        & (summary["setting"] == setting)
        & (summary["estimand"] == estimand)
    ]
    assert len(rows) == 1, (block, setting, estimand)
    return rows.iloc[0]


# ------------------------------------------------------------------- pass probabilities
def efficiency_band_pass(replicates: int, ratio: float = 1.0) -> float:
    """P(the 99% interval of an efficiency ratio lies inside the band), normal approximation."""
    low, high = properties.EFFICIENCY_RATIO_BAND
    spread = ratio / np.sqrt(2.0 * (replicates - 1))
    half = Z * spread
    return float(norm.cdf((high - half - ratio) / spread) - norm.cdf((low + half - ratio) / spread))


def calibration_cell(replicates: int, ratio: float = 1.0) -> float:
    """One calibration positive: the shared rule times the efficiency band (a lower bound)."""
    shared = float(sizing.calibration_pass(MARGINS, replicates, np.array([ratio]))[0])
    return shared * efficiency_band_pass(replicates)


def calibration_family(replicates: int) -> float:
    """The six positives at target, as a product (cells that share draws only raise it)."""
    return calibration_cell(replicates) ** len(properties.CALIBRATION_LABELS)


def calibration_correlations(labels: list[str]) -> np.ndarray:
    """The exact correlation of the labels' curves; a stacked label has its in-sample curve."""
    matrix = law.curve_correlations("S")
    index = {curve: i for i, curve in enumerate(law.CURVES)}

    def curve(label: str) -> int:
        estimand, setting = properties.LABEL_SETTINGS[label]
        return index[f"{estimand}_{law.SETTINGS[setting].form}"]

    return np.array([[matrix[curve(a), curve(b)] for b in labels] for a in labels])


def pilot_ratios(summary: pd.DataFrame) -> dict[str, float]:
    return {
        f"{label}__correctly_specified": float(
            reading(summary, "calibration", setting, estimand)["se_ratio"]
        )
        for label, (estimand, setting) in properties.LABEL_SETTINGS.items()
    }


def pooled_family(summary: pd.DataFrame, replicates: int) -> float:
    """The arm-pooled predictive pass probability of the six positives at the pilot."""
    groups = sizing.calibration_groups(
        pilot_ratios(summary), pilot.CALIBRATION_DRAWS, calibration_correlations
    )
    tau = sizing.heterogeneity_bound(groups)
    return sizing.calibration_family_pass_probability(MARGINS, replicates, groups, tau)


def noise_control_probability(replicates: int) -> float:
    """P(the noise control's empirical efficiency interval lies above the band)."""
    ratio = float(np.sqrt(2.0))
    spread = ratio / np.sqrt(2.0 * (replicates - 1))
    high = properties.EFFICIENCY_RATIO_BAND[1]
    return float(norm.cdf((ratio - Z * spread - high) / spread))


def gain_se(correlation: float, replicates: int) -> float:
    """The SD of the log paired SD ratio of two normal estimates with this correlation."""
    return float(np.sqrt((1.0 - correlation**2) / replicates))


def gain_pass(ratio: float, correlation: float, replicates: int) -> float:
    """P(the 99% upper end of the paired ratio lies below the boundary)."""
    se = gain_se(correlation, replicates)
    return float(norm.cdf((np.log(properties.GAIN_BOUNDARY) - np.log(ratio)) / se - Z))


def gain_control_holds(ratio: float, correlation: float, replicates: int) -> float:
    """P(the 99% upper end of a zero-gain pair reaches the boundary)."""
    return 1.0 - gain_pass(ratio, correlation, replicates)


def se_sanity_pass(ratio: float, replicates: int) -> float:
    """P(the point SE ratio lies in the sanity band), normal approximation."""
    low, high = MARGINS.se_ratio_sanity
    spread = ratio / np.sqrt(2.0 * (replicates - 1))
    return float(norm.cdf((high - ratio) / spread) - norm.cdf((low - ratio) / spread))


def coverage_lower(row: pd.Series) -> float:
    """The one-sided 80% lower Clopper-Pearson bound of a pilot coverage."""
    from scipy.stats import beta

    draws = int(row["draws"])
    covered = round(float(row["coverage"]) * draws)
    return float(beta.ppf(1.0 - sizing.PILOT_LEVEL, covered, draws - covered + 1))


def accuracy_cell(row: pd.Series, replicates: int) -> float:
    """The coverage floor and SE band of one accuracy positive at its pilot reading."""
    return sizing.coverage_floor_pass_probability(
        MARGINS, replicates, coverage_lower(row)
    ) * se_sanity_pass(float(row["se_ratio"]), replicates)


#: Each accuracy cell's pilot block and setting.
ACCURACY_PILOT = {
    cell: (
        {
            "s_wrong_known_score": "accuracy_s",
            "b2_wrong_known_score": "accuracy_b2",
            "b3_wrong_known_score": "accuracy_b3",
            "s_declared_constant": "control",
        }[setting],
        setting,
        name,
    )
    for cell, (setting, name, _) in properties.ACCURACY_SETTINGS.items()
}
POSITIVE_ACCURACY = tuple(
    cell for cell, (_, _, role) in properties.ACCURACY_SETTINGS.items() if role == "positive"
)


def accuracy_correlations(labels: list[str]) -> np.ndarray:
    """Cells of one draw kind share each draw (correlation one, the cautious reading)."""
    kinds = [label.split("_", 1)[0] for label in labels]
    return np.array([[1.0 if a == b else 0.0 for b in kinds] for a in kinds])


def accuracy_biases(summary: pd.DataFrame) -> dict[str, float]:
    return {
        f"{properties.ACCURACY}/{cell}": float(
            reading(summary, *ACCURACY_PILOT[cell])["bias_over_sd"]
        )
        for cell in POSITIVE_ACCURACY
    }


def rate_slope_pass(slope: float, replicates: int, sizes: tuple[int, ...]) -> float:
    """P(the 99% slope interval of the log SD on log n lies in the root-n band, clear of -1/4).

    Each rung's log SD has variance ``1 / (2 (R - 1))``, and the slope is least squares.
    """
    x = np.log(np.asarray(sizes, dtype=float))
    se = float(np.sqrt(1.0 / (2.0 * (replicates - 1)) / np.sum((x - x.mean()) ** 2)))
    low = property_verdicts.ROOT_N_SLOPE - property_verdicts.ROOT_N_SLOPE_MARGIN
    high = property_verdicts.ROOT_N_SLOPE + property_verdicts.ROOT_N_SLOPE_MARGIN
    high = min(high, property_verdicts.EXCLUDED_SLOPE)
    return float(norm.cdf((high - Z * se - slope) / se) - norm.cdf((low + Z * se - slope) / se))


def designed_pass(summary: pd.DataFrame) -> dict[str, float]:
    """Each gated instrument's pass probability at the declared sizes, and the red chance."""
    k, a, r = (
        properties.CALIBRATION_REPLICATES,
        properties.ACCURACY_REPLICATES,
        properties.RATE_REPLICATES,
    )
    out: dict[str, float] = {}
    out["calibration"] = (
        pooled_family(summary, k)
        * efficiency_band_pass(k) ** len(properties.CALIBRATION_LABELS)
        * (
            sizing.calibration_control_probability(MARGINS, k, properties.SHRUNKEN_SE_FACTOR)
            * noise_control_probability(k)
        )
        ** len(properties.CALIBRATION_LABELS)
    )
    gain = 1.0
    for estimand in ("att", "atc"):
        ratio = law.curve_correlation("S", estimand)
        row = reading(summary, "zero_gain", "s0_known_score", estimand)
        gain *= gain_pass(ratio, ratio, k) * gain_control_holds(
            float(row["paired_sd_ratio"]), float(row["paired_correlation"]), k
        )
    out["efficiency_gain"] = gain
    out["accuracy"] = accuracy_family(summary, a)
    out["root_n"] = root_n_pass(summary, r)
    primary = 1.0
    for estimand in study.PRIMARY_ESTIMANDS:
        row = reading(summary, "calibration", "s_known_score", estimand)
        primary *= sizing.bias_equivalent_probability(
            MARGINS, float(row["bias_over_sd"]), study.PRIMARY_REPLICATES
        ) * sizing.coverage_floor_pass_probability(
            MARGINS, study.PRIMARY_REPLICATES, coverage_lower(row)
        )
    out["primary"] = float(primary)
    out["red"] = 1.0 - float(np.prod(list(out.values())))
    return out


def accuracy_family(summary: pd.DataFrame, replicates: int) -> float:
    """Every accuracy positive's three clauses and the control's discrimination."""
    bias = sizing.property_bias_pass_probability(
        MARGINS,
        accuracy_biases(summary),
        pilot.DRAWS,
        {properties.ACCURACY: replicates},
        accuracy_correlations,
    )
    rest = np.prod(
        [accuracy_cell(reading(summary, *ACCURACY_PILOT[c]), replicates) for c in POSITIVE_ACCURACY]
    )
    control = reading(summary, "control", "s_declared_constant", "att")
    gap = float(control["bias_over_sd"])
    return float(bias * rest * sizing.bias_discriminated_probability(MARGINS, gap, replicates))


def property_size(pass_at: Callable[[int], float]) -> int:
    """The smallest multiple of 200, at least the primaries' 1,000, with pass at least the family share.

    The coverage clause reads each cell at the one-sided 80% lower bound of its pilot
    coverage, as clause (c) of :func:`tests.studies.repeated_sizing.primary_replicates` does.
    """
    return next(
        r for r in range(1_000, 20_001, sizing.PROPERTY_STEP) if pass_at(r) >= PROPERTY_FLOOR
    )


def rate_readings(summary: pd.DataFrame) -> dict[int, pd.Series]:
    return {
        500: reading(summary, "rate_500", "s_known_score", "att"),
        2_000: reading(summary, "calibration", "s_known_score", "att"),
        8_000: reading(summary, "rate_8000", "s_known_score", "att"),
    }


def rate_slope(summary: pd.DataFrame) -> float:
    rows = rate_readings(summary)
    x = np.log(np.array(properties.RATE_SIZES, dtype=float))
    y = np.log(np.array([float(rows[size]["sd"]) for size in properties.RATE_SIZES]))
    return float(np.polyfit(x, y, 1)[0])


def root_n_pass(summary: pd.DataFrame, replicates: int) -> float:
    """Every rung's three clauses and both slopes (the reported slope read as the spread's)."""
    rungs = np.prod(
        [
            sizing.bias_equivalent_probability(MARGINS, float(row["bias_over_sd"]), replicates)
            * sizing.coverage_floor_pass_probability(MARGINS, replicates, coverage_lower(row))
            * se_sanity_pass(float(row["se_ratio"]), replicates)
            for row in rate_readings(summary).values()
        ]
    )
    slope = rate_slope_pass(rate_slope(summary), replicates, properties.RATE_SIZES)
    return float(rungs * slope**2)


@functools.cache
def exact_sd(name: str, estimand: str, form: str) -> float:
    return law.efficiency_sd(name, estimand, form)


# --------------------------------------------------------------------------- the tests
class TestTheDeclaration:
    def test_the_constants(self) -> None:
        assert study.SLUG == "known-score-att"
        assert (study.SEED, study.RESAMPLING_SEED) == (20263501, 20263502)
        assert study.STUDY.publication_policy == "reporting"
        assert study.STUDY.reference is None
        assert (study.PRIMARY_N, study.PRIMARY_REPLICATES) == (2_000, 2_400)
        assert properties.CALIBRATION_REPLICATES == properties.ZERO_GAIN_REPLICATES == 4_800
        assert (properties.ACCURACY_REPLICATES, properties.RATE_REPLICATES) == (1_400, 1_600)
        assert properties.RATE_SIZES == (500, 2_000, 8_000)
        assert (law.Q_BOUNDS, law.N_FOLDS, law.MAX_ITER, law.TOL) == ((-8.0, 8.0), 5, 100, 1e-10)
        assert law.CONTROL_CONSTANT == 0.49
        assert properties.GAIN_BOUNDARY == 0.98
        assert properties.EFFICIENCY_RATIO_BAND == (0.90, 1.10)

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
        built["root_n_rate"] = {"empirical_sd", "reported_se"}
        assert built == {family: set(cells) for family, cells in study.PROPERTY_CELLS.items()}

    def test_the_roles(self) -> None:
        roles = {(c.property, c.cell): c.role for c in properties.declared_cells()}
        for label in properties.CALIBRATION_LABELS:
            assert roles[("interval_calibration", f"{label}__correctly_specified")] == "positive"
            for kind in ("shrunken_se_control", "noise_control"):
                assert roles[("interval_calibration", f"{label}__{kind}")] == "control"
        for label in properties.GAIN_LABELS:
            expected = "control" if label.endswith("zero_gain") else "positive"
            assert roles[("efficiency_gain", f"{label}__known_score")] == expected
            assert roles[("efficiency_gain", f"{label}__arm_mean")] == (
                property_verdicts.DIAGNOSTIC_ROLE
            )
        assert roles[("known_score_accuracy", "s_att__declared_constant")] == "control"
        for cell in POSITIVE_ACCURACY:
            assert roles[("known_score_accuracy", cell)] == "positive"
        for size in properties.RATE_SIZES:
            assert roles[("root_n_and_efficiency", f"n_{size}")] == "positive"

    def test_the_cells_fit_the_declared_option(self) -> None:
        settings = law.SETTINGS
        for label, (_, setting) in properties.LABEL_SETTINGS.items():
            form = "arm_mean" if "arm_mean" in label else "known_score"
            assert settings[setting].form == form
            assert settings[setting].cross_fit == label.endswith("stacked")
            assert settings[setting].outcome == "correct" and settings[setting].law == "S"
        for label, (_, numerator, denominator) in properties.GAIN_SETTINGS.items():
            assert settings[numerator].form == "known_score"
            assert settings[denominator].form == "arm_mean"
            assert settings[numerator].law == settings[denominator].law
            assert settings[numerator].law == ("S0" if label.endswith("zero_gain") else "S")
        for setting, _ in properties.ACCURACY_KINDS.values():
            assert settings[setting].form == "known_score"
        assert settings["s_declared_constant"].declared == law.CONTROL_CONSTANT
        assert settings["s_declared_constant"].outcome == "correct"

    def test_each_kind_draws_its_own_named_stream(self) -> None:
        seeds: dict[str, list[int]] = {}
        for ((kind, argument),) in properties.payloads(budget=2):
            if kind in ("accuracy", "rate"):
                stream, replicate = (
                    argument[0] if kind == "accuracy" else f"rate_n_{argument[0]}",
                    argument[1],
                )
            else:
                stream, replicate = kind, argument
            seeds.setdefault(stream, []).append(properties._seed(stream, replicate))
        # Calibration, zero gain, four accuracy kinds (the control among them), three rungs.
        assert len(seeds) == 9
        assert len({tuple(values) for values in seeds.values()}) == 9

    def test_the_paired_cells_share_their_draws(self) -> None:
        """The gain pairs and the in-sample calibration labels are fits of one draw."""
        key = (("calibration", 0),)
        rows = properties._dispatch(key[0], properties._dispatch_draw(key[0]))
        by_cell = {(row["property"], row["cell"]): row["estimate"] for row in rows}
        for estimand in ("att", "atc"):
            for form in ("known_score", "arm_mean"):
                assert (
                    by_cell[("efficiency_gain", f"{estimand}__{form}")]
                    == by_cell[
                        (
                            "interval_calibration",
                            f"{estimand}_{form}_in_sample__correctly_specified",
                        )
                    ]
                )
        assert set(by_cell) | {
            name for names in properties.DERIVED_CELLS.values() for name in names
        } >= {tuple(name.split("/", 1)) for name in properties._cells_of(key)} - {
            tuple(name.split("/", 1))
            for names in properties.DERIVED_CELLS.values()
            for name in names
        }


class TestTheLaws:
    def test_the_truths_converge_and_match_monte_carlo(self) -> None:
        exact = law.truth("S")
        fine = law.truth("S", 400)
        for name in ("p1", "ate", "att", "atc"):
            assert fine[name] == pytest.approx(exact[name], abs=1e-12)
        rng = np.random.default_rng(11)
        w1 = rng.uniform(-law.ROOT3, law.ROOT3, 4_000_000)
        w2 = rng.binomial(1, 0.5, 4_000_000).astype(float)
        g = law.g0(w1, w2)
        t = law.tau("S", w1)
        assert np.mean(g) == pytest.approx(exact["p1"], abs=1e-3)
        assert np.mean(g * t) / np.mean(g) == pytest.approx(exact["att"], abs=3e-3)
        assert np.mean((1 - g) * t) / np.mean(1 - g) == pytest.approx(exact["atc"], abs=3e-3)

    def test_the_declared_law_s_numbers(self) -> None:
        exact = law.truth("S")
        assert exact["p1"] == pytest.approx(0.489987, abs=1e-6)
        assert exact["ate"] == pytest.approx(1.0, abs=1e-12)
        assert exact["att"] == pytest.approx(1.663141, abs=1e-6)
        assert exact["atc"] == pytest.approx(0.362897, abs=1e-6)
        assert law.control_bias() == pytest.approx(-0.663141, abs=1e-6)
        assert float(law.g0(-law.ROOT3, 0.0)) == pytest.approx(0.115879, abs=1e-6)
        assert float(law.g0(law.ROOT3, 1.0)) == pytest.approx(0.873476, abs=1e-6)
        assert pytest.approx(exact["p1"], abs=0.005) == law.CONTROL_CONSTANT

    def test_the_g_range_is_inside_the_att_default_pair(self) -> None:
        assert float(law.g0(-law.ROOT3, 0.0)) > 0.025 and float(law.g0(law.ROOT3, 1.0)) < 0.975
        assert 0.025 < law.CONTROL_CONSTANT < 0.975

    def test_the_outcome_support_is_inside_the_declared_bounds(self) -> None:
        low, high = law.outcome_support("S")
        assert (low, high) == pytest.approx((-5.062178, 7.562178), abs=1e-6)
        assert law.Q_BOUNDS[0] < low and high < law.Q_BOUNDS[1]
        frame = law.draw("S", 200_000, 5)
        assert frame["Y"].between(low, high).all()

    def test_the_draws_have_the_law_s_moments(self) -> None:
        frame = law.draw("S", 400_000, 3)
        w1, w2, a = frame["W1"], frame["W2"], frame["A"]
        residual = frame["Y"] - law.qbar0("S", a, w1, w2)
        assert abs(residual.mean()) < 5e-3
        assert (residual**2).mean() == pytest.approx(law.NOISE_VARIANCE, rel=0.01)
        assert a.mean() == pytest.approx(law.truth("S")["p1"], abs=3e-3)
        assert (frame["p1"] == law.g0(w1, w2)).all()
        constant = law.draw("S", 1_000, 3, declared=law.CONTROL_CONSTANT)
        assert (constant["p1"] == law.CONTROL_CONSTANT).all()

    @pytest.mark.parametrize("estimand", ["att", "atc"])
    def test_the_curve_sds(self, estimand: str) -> None:
        """The closed forms against the curves sampled from their definitions."""
        rng = np.random.default_rng(17)
        n = 4_000_000
        w1 = rng.uniform(-law.ROOT3, law.ROOT3, n)
        w2 = rng.binomial(1, 0.5, n).astype(float)
        g = law.g0(w1, w2)
        a = (rng.uniform(size=n) < g).astype(float)
        eps = rng.uniform(-law.ROOT3, law.ROOT3, n)
        psi = law.truth("S")[estimand]
        t = law.tau("S", w1)
        if estimand == "att":
            share, weight = law.truth("S")["p1"], g
            residual = a * eps - (1 - a) * g / (1 - g) * eps
            arm = a
        else:
            share, weight = 1 - law.truth("S")["p1"], 1 - g
            residual = a * (1 - g) / g * eps - (1 - a) * eps
            arm = 1 - a
        known = (residual + weight * (t - psi)) / share
        arm_mean = (residual + arm * (t - psi)) / share
        assert known.std() == pytest.approx(exact_sd("S", estimand, "known_score"), rel=5e-3)
        assert arm_mean.std() == pytest.approx(exact_sd("S", estimand, "arm_mean"), rel=5e-3)

    def test_the_declared_curve_sds_and_ratios(self) -> None:
        for label, value in properties.EFFICIENCY_SD.items():
            estimand, setting = properties.LABEL_SETTINGS[label]
            form = law.SETTINGS[setting].form
            assert value == pytest.approx(exact_sd("S", estimand, form), abs=1e-6)
        for label, (known, arm_mean) in properties.GAIN_SD.items():
            name = "S0" if label.endswith("zero_gain") else "S"
            estimand = label.split("_", 1)[0]
            assert known == pytest.approx(exact_sd(name, estimand, "known_score"), abs=1e-6)
            assert arm_mean == pytest.approx(exact_sd(name, estimand, "arm_mean"), abs=1e-6)
        assert law.curve_correlation("S", "att") == pytest.approx(0.9014, abs=1e-4)
        assert law.curve_correlation("S", "atc") == pytest.approx(0.9081, abs=1e-4)
        for estimand in ("att", "atc"):
            assert law.curve_correlation("S0", estimand) == pytest.approx(1.0, abs=1e-12)
        assert study.STUDY.efficiency_bounds == pytest.approx(
            {
                "att": properties.EFFICIENCY_SD["att_known_score_in_sample"],
                "atc": properties.EFFICIENCY_SD["atc_known_score_in_sample"],
            },
            abs=1e-6,
        )

    def test_the_gain_is_the_projection_term(self) -> None:
        """``Var D_AM - Var D_KS = E[g (1 - g) (tau - psi)^2] / p^2`` (plan section 2.2)."""
        at = law.grid()
        g = law.g0(at.w1, at.w2)
        exact = law.truth("S")
        for estimand, share in (("att", exact["p1"]), ("atc", 1 - exact["p1"])):
            gap = at.mean(g * (1 - g) * (law.tau("S", at.w1) - exact[estimand]) ** 2) / share**2
            difference = (
                exact_sd("S", estimand, "arm_mean") ** 2
                - exact_sd("S", estimand, "known_score") ** 2
            )
            assert difference == pytest.approx(gap, rel=1e-10)

    def test_law_b_two_arm_truths_are_the_shipped_study_s(self) -> None:
        shipped = law_b.truth(2)
        ours = law.law_b_truth(2)
        assert ours["att"] == pytest.approx(shipped["att"], abs=1e-14)
        assert ours["atc"] == pytest.approx(shipped["atc"], abs=1e-14)

    def test_law_b_three_arm_truths_match_monte_carlo(self) -> None:
        rng = np.random.default_rng(23)
        n = 4_000_000
        w1 = rng.standard_normal(n)
        w2 = rng.binomial(1, 0.5, n).astype(float)
        w3 = rng.uniform(-1, 1, n)
        g = law_b.mechanism(w2, 3)
        q = [law_b.outcome_probability(arm, w1, w2, w3) for arm in range(3)]
        exact = law.law_b_truth(3)
        assert set(exact) == {
            "att[1.0 vs 0.0]",
            "att[2.0 vs 0.0]",
            "atc[1.0 vs 0.0]",
            "atc[2.0 vs 0.0]",
        }
        for arm in (1, 2):
            blip = q[arm] - q[0]
            att = np.mean(g[:, arm] * blip) / np.mean(g[:, arm])
            atc = np.mean(g[:, 0] * blip) / np.mean(g[:, 0])
            assert exact[f"att[{float(arm)} vs 0.0]"] == pytest.approx(att, abs=2e-3)
            assert exact[f"atc[{float(arm)} vs 0.0]"] == pytest.approx(atc, abs=2e-3)

    def test_the_correct_outcome_regression_is_the_law(self) -> None:
        frame = law.draw("S", 400_000, 9)
        learner = law.LawSOutcome("correct").fit(frame[["A", "W1", "W2"]].to_numpy(), frame["Y"])
        assert learner.model_.intercept_ == pytest.approx(0.0, abs=0.01)
        assert learner.model_.coef_ == pytest.approx([1.0, 1.0, 0.5, 1.5], abs=0.01)

    def test_the_wrong_outcome_regression_misses_the_effect_curve(self) -> None:
        frame = law.draw("S", 400_000, 9)
        learner = law.LawSOutcome("wrong").fit(frame[["A", "W1", "W2"]].to_numpy(), frame["Y"])
        design = frame[["A", "W1", "W2"]].to_numpy()
        one, zero = design.copy(), design.copy()
        one[:, 0], zero[:, 0] = 1.0, 0.0
        fitted = learner.predict(one) - learner.predict(zero)
        # The wrong regression's effect is a constant; the law's varies with W1 (SD 1.5).
        assert np.ptp(fitted) < 1e-9
        assert np.std(law.tau("S", frame["W1"]) - fitted) > 1.4


class TestTheSizes:
    def test_the_calibration_replications_at_target(self) -> None:
        floor = next(r for r in range(400, 20_001, 400) if calibration_family(r) >= FAMILY_FLOOR)
        assert floor == AT_TARGET
        assert calibration_family(floor) == pytest.approx(0.985, abs=2e-3)

    def test_the_pooled_predictive_rule(self, summary: pd.DataFrame) -> None:
        pooled = sizing.calibration_replicates(
            MARGINS,
            pilot_ratios(summary),
            pilot.CALIBRATION_DRAWS,
            calibration_correlations,
            family_pass=FAMILY_FLOOR,
        )
        assert max(AT_TARGET, pooled) == properties.CALIBRATION_REPLICATES
        family = pooled_family(summary, properties.CALIBRATION_REPLICATES)
        assert family * efficiency_band_pass(properties.CALIBRATION_REPLICATES) ** 6 >= FAMILY_FLOOR

    def test_the_calibration_pilot_reads_the_exact_correlations(
        self, summary: pd.DataFrame
    ) -> None:
        for estimand in ("att", "atc"):
            row = reading(summary, "calibration", "s_known_score", estimand)
            exact = calibration_correlations(
                [f"{estimand}_known_score_in_sample", f"{estimand}_arm_mean_in_sample"]
            )[0, 1]
            assert float(row["paired_correlation"]) == pytest.approx(exact, abs=0.01)

    def test_each_label_s_plug_in_pass_at_its_pilot_ratio(self, summary: pd.DataFrame) -> None:
        for label, (estimand, setting) in properties.LABEL_SETTINGS.items():
            row = reading(summary, "calibration", setting, estimand)
            plug_in = calibration_cell(properties.CALIBRATION_REPLICATES, float(row["se_ratio"]))
            assert plug_in >= PLUG_IN_FLOOR, (label, plug_in)

    def test_the_derived_controls_fail_with_certainty(self) -> None:
        r = properties.CALIBRATION_REPLICATES
        assert (
            sizing.calibration_control_probability(MARGINS, r, properties.SHRUNKEN_SE_FACTOR)
            > CERTAIN
        )
        assert noise_control_probability(r) > CERTAIN

    @pytest.mark.parametrize("estimand", ["att", "atc"])
    def test_the_paired_gate_passes_at_the_exact_ratio(self, estimand: str) -> None:
        ratio = law.curve_correlation("S", estimand)
        assert gain_pass(ratio, ratio, properties.CALIBRATION_REPLICATES) > CERTAIN

    @pytest.mark.parametrize("estimand", ["att", "atc"])
    def test_the_zero_gain_control_holds(self, summary: pd.DataFrame, estimand: str) -> None:
        row = reading(summary, "zero_gain", "s0_known_score", estimand)
        probability = gain_control_holds(
            float(row["paired_sd_ratio"]),
            float(row["paired_correlation"]),
            properties.ZERO_GAIN_REPLICATES,
        )
        assert probability > CERTAIN

    def test_a_boundary_at_one_would_leave_the_zero_gain_control_to_chance(
        self, summary: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Why the boundary is 0.98: at one, the control reads a term of order ``1 / n``."""
        monkeypatch.setattr(properties, "GAIN_BOUNDARY", 1.0)
        held = [
            gain_control_holds(
                float(reading(summary, "zero_gain", "s0_known_score", e)["paired_sd_ratio"]),
                float(reading(summary, "zero_gain", "s0_known_score", e)["paired_correlation"]),
                properties.ZERO_GAIN_REPLICATES,
            )
            for e in ("att", "atc")
        ]
        assert min(held) < 0.999

    def test_the_accuracy_replications(self, summary: pd.DataFrame) -> None:
        """The property bias rule, then the coverage floor and SE band at the pilot readings."""
        sizes = sizing.property_bias_replicates(
            MARGINS, accuracy_biases(summary), pilot.DRAWS, accuracy_correlations
        )
        assert sizes[properties.ACCURACY] <= properties.ACCURACY_REPLICATES
        size = property_size(lambda r: accuracy_family(summary, r))
        assert size == properties.ACCURACY_REPLICATES
        assert accuracy_family(summary, size) >= PROPERTY_FLOOR

    def test_the_control_is_discriminated(self, summary: pd.DataFrame) -> None:
        row = reading(summary, "control", "s_declared_constant", "att")
        exact_gap = law.control_bias() / float(row["sd"])
        assert exact_gap < -10.0
        for gap in (exact_gap, float(row["bias_over_sd"])):
            assert (
                sizing.bias_discriminated_probability(MARGINS, gap, properties.ACCURACY_REPLICATES)
                > CERTAIN
            )

    def test_the_root_n_ladder(self, summary: pd.DataFrame) -> None:
        assert rate_slope(summary) == pytest.approx(-0.5, abs=0.06)
        size = property_size(lambda r: root_n_pass(summary, r))
        assert size == properties.RATE_REPLICATES
        assert root_n_pass(summary, size) >= PROPERTY_FLOOR

    def test_the_primary_rule(self, summary: pd.DataFrame) -> None:
        covered = {
            estimand: round(
                float(reading(summary, "calibration", "s_known_score", estimand)["coverage"])
                * pilot.CALIBRATION_DRAWS
            )
            for estimand in study.PRIMARY_ESTIMANDS
        }
        replicates = sizing.primary_replicates(
            MARGINS, covered, pilot.CALIBRATION_DRAWS, implementations=1
        )
        assert replicates == study.PRIMARY_REPLICATES

    def test_the_designed_chance_of_a_red_cell(self, summary: pd.DataFrame) -> None:
        """One minus the product of every gated instrument's pass probability."""
        assert designed_pass(summary)["red"] == pytest.approx(DESIGNED_RED, abs=0.005)


class TestThePreRead:
    def test_the_pilot_ran_its_draws_from_draw_zero(self, summary: pd.DataFrame) -> None:
        for block, (_, settings, estimands) in pilot.BLOCKS.items():
            draws = pilot.BLOCK_DRAWS[block]
            for setting in settings:
                for estimand in estimands:
                    row = reading(summary, block, setting, estimand)
                    assert (int(row["draws"]), int(row["first_draw"])) == (draws, 0)
                    assert int(row["last_draw"]) == draws - 1

    def test_the_calibration_pilot_is_the_shared_pilot_size(self) -> None:
        assert pilot.CALIBRATION_DRAWS == sizing.CALIBRATION_PILOT_DRAWS

    @pytest.mark.parametrize("label", properties.CALIBRATION_LABELS)
    def test_each_calibration_label_reads_in_band(self, summary: pd.DataFrame, label: str) -> None:
        estimand, setting = properties.LABEL_SETTINGS[label]
        row = reading(summary, "calibration", setting, estimand)
        assert row["coverage"] >= PRE_READ_COVERAGE
        draws = int(row["draws"])
        # Five Monte Carlo errors of an SE ratio, 1 / sqrt(2 (R - 1)) relative.
        assert row["se_ratio"] == pytest.approx(1.0, abs=5 / np.sqrt(2 * (draws - 1)))
        efficiency = float(row["sd"]) * np.sqrt(int(row["n"])) / properties.EFFICIENCY_SD[label]
        assert efficiency == pytest.approx(1.0, abs=5 / np.sqrt(2 * (draws - 1)))

    @pytest.mark.parametrize("estimand", ["att", "atc"])
    def test_the_paired_ratio_reads_near_its_exact_value(
        self, summary: pd.DataFrame, estimand: str
    ) -> None:
        row = reading(summary, "calibration", "s_known_score", estimand)
        ratio = law.curve_correlation("S", estimand)
        se = gain_se(ratio, int(row["draws"]))
        assert np.log(float(row["paired_sd_ratio"]) / ratio) == pytest.approx(0.0, abs=5 * se)
        assert float(row["paired_correlation"]) == pytest.approx(ratio, abs=0.03)

    @pytest.mark.parametrize("estimand", ["att", "atc"])
    def test_the_zero_gain_pair_reads_no_gain(self, summary: pd.DataFrame, estimand: str) -> None:
        row = reading(summary, "zero_gain", "s0_known_score", estimand)
        assert abs(np.log(float(row["paired_sd_ratio"]))) < 0.002
        assert float(row["paired_correlation"]) > 0.999

    @pytest.mark.parametrize("cell", POSITIVE_ACCURACY)
    def test_each_accuracy_positive_reads_below_the_ceiling(
        self, summary: pd.DataFrame, cell: str
    ) -> None:
        row = reading(summary, *ACCURACY_PILOT[cell])
        assert abs(row["bias_over_sd"]) <= PILOT_BIAS_CEILING
        low, high = MARGINS.se_ratio_sanity
        assert low < row["se_ratio"] < high
        assert row["coverage"] >= PRE_READ_COVERAGE

    def test_the_control_reads_near_its_limit(self, summary: pd.DataFrame) -> None:
        row = reading(summary, "control", "s_declared_constant", "att")
        assert float(row["bias"]) == pytest.approx(
            law.control_bias(), abs=5 * float(row["sd"]) / 40
        )


#: The cells whose red verdict gates the study: every declared cell that is not diagnostic,
#: and the two rate slopes.
GATED = (
    *(
        f"{cell.property}/{cell.cell}"
        for cell in properties.declared_cells()
        if cell.role != property_verdicts.DIAGNOSTIC_ROLE
    ),
    "root_n_rate/empirical_sd",
    "root_n_rate/reported_se",
)


class TestTheRedCellReading:
    def test_the_gated_cells(self) -> None:
        # Six calibration positives with two controls each, two gain positives and two gain
        # controls, eight accuracy positives and one control, three rungs and two slopes.
        assert len(GATED) == 18 + 4 + 9 + 3 + 2
        assert len(set(GATED)) == len(GATED)

    def test_no_gated_cell_reads_an_inadmissible_red(self) -> None:
        """The declared red-cell reading, applied mechanically.

        Empty until the run commits its tables; at registration a red control, or a red
        positive without its finite-sample owner, fails here.
        """
        from tests.studies.mtp_study_tools import gating_findings

        assert gating_findings(study.STUDY, GATED, truth_implementations=("cleverly",)) == []
