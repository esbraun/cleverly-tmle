"""The declared design of ``longitudinal-repeated-crossfit`` (``X46-plan.md`` 5.4).

Every number here was fixed before any registered run.  The readings come from the sizing
probe ``tests/diagnostics/x46_sizing/`` (``reading.csv``), which fits the study's own declared
repeated fits on seeds outside every declared stream.

* **Truths.**  Every truth is a finite sum on an exact law.  Law M's duration is centred at
  :data:`~tests.studies.longitudinal_repeated_crossfit.M_STAR`, which makes the exact intercept
  and slope curves uncorrelated; the slope is the uncentred slope and the intercept is
  :math:`b_0 + m^* b_1`.
* **Bounds.**  ``g_bounds = (0.01, 1)`` never binds at the truth on laws D, S, C and I, and the
  fallback-read bias bound of law I is below 0.1 standard errors at n = 40,000.
* **The sizing rule** (X19's projection, applied to X46's replication counts).  A pilot reading
  at ``n0`` projects its standardized bias as :math:`(|b| + m)\\sqrt{n_0 / n}` and its SE-ratio
  deficit as :math:`(1 - \\min(r, 1)) n_0 / n`.  Every primary row passes its bias rule with
  probability at least 0.95 at 1,600 replications.  Every calibration cell passes with
  probability at least 0.90 at 6,000, and the nine pass jointly, as the independent product,
  with probability at least 0.95.  3,600 and 4,800 replications fail the family rule.
* **Controls.**  Every band label's pointwise control and the shrunken-SE controls fail with
  power at least 0.90.  The ``repeat_increment`` gate passes with probability at least 0.95,
  at the reading and predictively, and its inflated-SE control has power at least 0.90.
* **Predictive probabilities** (rule 7).  Each pass probability is also averaged over a normal
  draw of the reading about its value with its Monte Carlo SE.  :func:`declaration` prints both.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom, norm
from scipy.stats import t as student

from tests import discrete_law_longitudinal as dlaw
from tests import discrete_law_longitudinal_incremental as ilaw
from tests.studies import longitudinal_incremental_common as common
from tests.studies import longitudinal_msm_properties as msm_law
from tests.studies import longitudinal_repeated_crossfit as s
from tests.studies import longitudinal_repeated_crossfit_properties as p
from tests.studies.canonical_full_refit_bootstrap import calibration_pass_probability
from tests.studies.evidence import clopper_pearson
from tests.studies.evidence.registry import ROOT, Margins, registered
from tests.studies.evidence.seeds import replicate_seed

pytestmark = pytest.mark.xdist_group("longitudinal_repeated_crossfit_design")

READING = pd.read_csv(ROOT / "tests" / "diagnostics" / "x46_sizing" / "reading.csv")
MARGINS = Margins()
PRIMARY_PASS = 0.95
CALIBRATION_PASS = 0.90
FAMILY_PASS = 0.95
CONTROL_POWER = 0.90
INCREMENT_PASS = 0.95
#: The order of the Gauss-Hermite rule of every predictive probability.
NODES = 24
#: Two-sided 99% normal quantile: the half-width multiplier of every bootstrap interval.
Z99 = float(norm.ppf(0.5 + MARGINS.confidence_level / 2.0))


def test_the_declared_numbers() -> None:
    assert s.STUDY.publication_policy == "reporting"
    assert s.STUDY.reference is None
    assert s.STUDY.margins == MARGINS
    assert (s.PRIMARY_N, s.PRIMARY_REPLICATES, s.IPSI_N) == (8_000, 1_600, 40_000)
    assert (s.N_FOLDS, s.LEARNER_FOLDS, s.REPEATS, s.G_BOUNDS) == (5, 2, 3, (0.01, 1.0))
    assert (s.SEED, s.RESAMPLING_SEED) == (20261446, 2026144601)
    assert (p.CALIBRATION_N, p.CALIBRATION_REPLICATES) == (8_000, 6_000)
    assert (p.STABILITY_TRIALS, p.RATE_REPLICATES, p.RATE_SIZES) == (
        400,
        800,
        (2_000, 8_000, 32_000),
    )
    assert p.IPSI_PRIMARY_REPLICATES == 1_600
    assert (p.SHRUNKEN_SE_FACTOR, p.INFLATED_SE_FACTOR) == (0.70, 1.05)
    assert (p.INCREMENT_CEILING, p.INCREMENT_HORIZON, p.MAX_REPEAT_SPREAD_RATIO) == (0.03, 3, 1.0)
    assert p.FAILURE_BOUND == 0.01


def test_the_study_draws_from_its_own_seeds() -> None:
    mine = {s.SEED, s.RESAMPLING_SEED}
    others = {
        seed
        for record in registered()
        if record.slug != s.STUDY.slug
        for seed in (record.seed, record.resampling_seed)
    }
    assert len(mine) == 2 and not mine & others


def test_every_sample_stream_is_its_own() -> None:
    """The primary, property and stability streams share no sample seed."""
    seeds = [
        replicate_seed(s.STUDY, scenario, r)
        for scenario in s.SCENARIOS
        if scenario not in s.STUDY.scenario_seed_owners
        for r in range(200)
    ]
    seeds += [p.sample_seed(fit_set, r) for fit_set in p.FIT_SETS for r in range(200)]
    assert len(set(seeds)) == len(seeds)


# ------------------------------------------------------------------------------- truths


def test_the_law_d_truths_are_the_closed_forms() -> None:
    for name in (*s.D_MEANS, *s.D_CONTRASTS.values()):
        assert s.D_TRUTH[name] == pytest.approx(dlaw.TRUTH[name], abs=1e-12)
    ratio = dlaw.TRUTH["ey_regimen[always]"] / dlaw.TRUTH["ey_regimen[never]"]
    assert s.D_TRUTH[s.D_RATIO] == pytest.approx(ratio, abs=1e-12)
    assert abs(s.D_TRUTH[s.D_CONTRASTS["static"]]) > 0.05


def test_the_centre_decorrelates_the_exact_coefficient_curves() -> None:
    """A nonzero witness: the uncentred curves correlate strongly, the centred ones not at all."""
    curves = msm_law.influence_curves()

    def correlation(intercept: np.ndarray) -> float:
        covariance = float(np.sum(dlaw.PROBS * intercept * curves[:, 1]))
        scale = np.sqrt(np.sum(dlaw.PROBS * intercept**2) * np.sum(dlaw.PROBS * curves[:, 1] ** 2))
        return covariance / float(scale)

    assert correlation(curves[:, 0]) < -0.9
    assert abs(correlation(curves[:, 0] + s.M_STAR * curves[:, 1])) < 1e-12
    assert abs(s.M_STAR - 1.2304208) < 1e-7


def test_the_centred_model_shifts_only_the_intercept() -> None:
    b0 = msm_law.TRUTH["msm_regimen[(intercept)]"]
    b1 = msm_law.TRUTH["msm_regimen[duration]"]
    assert s.M_TRUTH["msm_regimen[duration]"] == pytest.approx(b1, abs=1e-12)
    assert s.M_TRUTH["msm_regimen[(intercept)]"] == pytest.approx(b0 + s.M_STAR * b1, abs=1e-12)
    assert s.M_TRUTH["msm_regimen[(intercept)]"] == pytest.approx(0.484322, abs=1e-6)
    assert s.M_TRUTH["msm_regimen[duration]"] == pytest.approx(0.180122, abs=1e-6)


def test_the_incremental_truths_are_the_oracle() -> None:
    for name in s.I_NAMES:
        assert s.I_TRUTH[name] == pytest.approx(common.truth(ilaw.LAW_A, s.I_PLANS, name))
    for name in s.S_NAMES:
        assert s.S_TRUTH[name] == pytest.approx(common.truth(ilaw.LAW_D, s.ARMS, name))
    for name in s.C_NAMES:
        assert s.C_TRUTH[name] == pytest.approx(common.truth(ilaw.LAW_CK.marginal, s.ARMS, name))
    assert abs(s.I_TRUTH[s.I_CONTRASTS["contrast_x2"]]) > 0.05


# ------------------------------------------------------------------------------- bounds


def law_d_cumulative(arms: tuple[Any, Any]) -> float:
    """The smallest true cumulative treatment-and-retention probability of one law-D plan."""
    smallest = 1.0
    for w in range(2):
        a1 = dlaw._arm(arms[0], w)
        first = (dlaw.G1[w] if a1 else 1.0 - dlaw.G1[w]) * dlaw.C1[w, a1]
        for l2 in range(2):
            a2 = dlaw._arm(arms[1], w, l2)
            g2 = dlaw.G2[w, a1, l2]
            second = (g2 if a2 else 1.0 - g2) * dlaw.C2[w, a1, l2, a2]
            smallest = min(smallest, float(first * second))
    return smallest


def oracle_cumulative(target: Any, level: int) -> float:
    """The smallest true cumulative probability of the static plan ``level`` on an oracle law.

    The walk follows every history with positive mass.  A treatment node multiplies by the
    probability of ``level`` given the history, a censoring node by the probability of staying
    observed; an event ends the history.
    """
    probs = np.asarray(target.probs, dtype=float)
    axes = target.axes
    found: list[float] = []

    def walk(prefix: tuple[int, ...], product: float, treated: bool) -> None:
        position = len(prefix)
        if position == len(axes):
            found.append(product)
            return
        upper = probs.sum(axis=tuple(range(position + 1, len(axes))))[prefix]
        total = float(upper.sum())
        if total <= 0:
            return
        kind = axes[position].kind
        if kind == "treatment":
            walk((*prefix, level), product * float(upper[level]) / total, True)
        elif kind == "censoring":
            walk((*prefix, 1), product * float(upper[1]) / total, treated)
        elif kind == "event":
            found.append(product)
            walk((*prefix, 0), product, treated)
        else:
            for value in range(axes[position].levels):
                if upper[value] > 0:
                    walk((*prefix, value), product, treated)

    walk((), 1.0, False)
    return min(found)


def test_the_lower_bound_never_binds_on_law_d() -> None:
    """The smallest cumulative product is ``always``'s 0.0352, 3.5 times the bound."""
    labels = (*s.D_REGIMENS, *msm_law.LABELS)
    smallest = {label: law_d_cumulative(dlaw.REGIMEN_ARMS[label]) for label in labels}
    assert smallest["always"] == pytest.approx(0.03515625, abs=1e-12)
    assert min(smallest.values()) > 3 * s.G_BOUNDS[0]


@pytest.mark.parametrize(
    "target",
    [ilaw.LAW_A, ilaw.LAW_D, ilaw.LAW_CK.marginal],
    ids=["law_i", "law_s", "law_c"],
)
def test_the_lower_bound_never_binds_on_the_incremental_laws(target: Any) -> None:
    """Both static arms keep every cumulative product above three times the bound.

    Law I's tilts sit between the two arms, so its arms bound every tilted product too.
    """
    for level in (0, 1):
        assert oracle_cumulative(target, level) > 3 * s.G_BOUNDS[0]


def test_law_i_meets_the_fallback_bound() -> None:
    bound = common.fallback_bound(ilaw.LAW_A, s.I_PLANS, s.I_NAMES, n=s.IPSI_N, n_folds=s.N_FOLDS)
    assert bound < common.FALLBACK_BOUND


# -------------------------------------------------------------------- the sizing rule


def probe(cell_set: str, scenario: str, name: str, reading: str = "median") -> Any:
    rows = READING[
        (READING["set"] == cell_set)
        & (READING["scenario"] == scenario)
        & (READING["name"] == name)
        & (READING["reading"] == reading)
    ]
    (row,) = rows.itertuples()
    return row


def projected(row: Any, n: int) -> tuple[float, float]:
    """The standardized bias and the SE ratio at ``n``, from the probe at ``row.n``.

    Adapted from ``projected`` in ``tests/unit/test_longitudinal_incremental_design.py`` on the
    X19 branch (``e288b730``): the bias falls as :math:`1/\\sqrt n` after adding one Monte Carlo
    SE, and an SE-ratio deficit falls as :math:`1/n`.  X19 reads a ratio above one as one.
    Here a ratio above one keeps its reading (``X46-plan.md`` 5.4 rule 1: it is not projected),
    so an over-covering cell can fail the calibration band.
    """
    bias = (abs(row.standardized_bias) + row.mcse) * np.sqrt(row.n / n)
    return float(bias), project_ratio(row.se_ratio, row.n, n)


def project_ratio(ratio: float, n0: int, n: int) -> float:
    """An SE ratio read at ``n0``, at ``n``: a deficit falls as 1/n, an excess stays."""
    return float(ratio if ratio >= 1.0 else 1.0 - (1.0 - ratio) * n0 / n)


def bias_pass_probability(bias: float, replicates: int) -> float:
    """The chance the 99% Student interval of the mean error lies inside the bias margin.

    A copy of the X19 helper of the same name (``e288b730``).
    """
    half = float(student.ppf(0.5 + MARGINS.confidence_level / 2.0, replicates - 1))
    spread = 1.0 / np.sqrt(replicates)
    room = MARGINS.standardized_bias - half * spread
    return float(norm.cdf((room - bias) / spread) - norm.cdf((-room - bias) / spread))


def calibrated_at(bias: float, ratio: float, replicates: int) -> float:
    """The calibration pass probability of a cell with this bias and SE ratio."""
    critical = float(norm.ppf(0.975)) * ratio
    coverage = float(norm.cdf(critical - bias) - norm.cdf(-critical - bias))
    return calibration_pass_probability(
        replicates,
        true_se_ratio=ratio,
        coverage=coverage,
        confidence_level=MARGINS.confidence_level,
        se_band=MARGINS.calibration_se_ratio,
        coverage_band=MARGINS.calibration_coverage,
    )


def calibrated_pass_probability(row: Any, n: int, replicates: int) -> float:
    return calibrated_at(*projected(row, n), replicates)


def ratio_mcse(row: Any) -> float:
    """The Monte Carlo SE of an SE-ratio reading, from the spread's relative variance."""
    return float(row.se_ratio / np.sqrt(2.0 * (row.replicates - 1)))


def predictive(probability: Callable[..., float], *centres: tuple[float, float]) -> float:
    """``probability`` averaged over independent normal draws of its arguments.

    Each centre is ``(value, Monte Carlo SE)``; a Gauss-Hermite rule of :data:`NODES` points per
    argument does the integral.
    """
    nodes, weights = np.polynomial.hermite_e.hermegauss(NODES)
    weights = weights / weights.sum()
    total = 0.0
    for index in np.ndindex(*(NODES,) * len(centres)):
        weight = float(np.prod([weights[i] for i in index]))
        values = [mean + nodes[i] * sd for i, (mean, sd) in zip(index, centres, strict=True)]
        total += weight * probability(*values)
    return total


def predictive_bias(row: Any, n: int, replicates: int) -> float:
    def at(bias: float) -> float:
        return bias_pass_probability(abs(bias) * np.sqrt(row.n / n), replicates)

    return predictive(at, (row.standardized_bias, row.mcse))


def predictive_calibration(row: Any, n: int, replicates: int) -> float:
    def at(bias: float, ratio: float) -> float:
        return calibrated_at(
            abs(bias) * np.sqrt(row.n / n), project_ratio(ratio, row.n, n), replicates
        )

    return predictive(at, (row.standardized_bias, row.mcse), (row.se_ratio, ratio_mcse(row)))


#: The probe set of each scenario at the declared size.
SETS = {
    "law_d_ltmle": "d@8000",
    "law_d_sdr": "d@8000",
    "law_s": "s@8000",
    "law_c": "c@8000",
    "law_m": "m@8000",
    "law_i": f"i@{s.IPSI_N}",
}
PRIMARY_ROWS = (
    *(
        (scenario, name, s.PRIMARY_N, s.PRIMARY_REPLICATES)
        for scenario, names in s.SCENARIOS.items()
        for name in names
    ),
    *(
        ("law_i", name, s.IPSI_N, p.IPSI_PRIMARY_REPLICATES)
        for name in s.IPSI_PRIMARY_LABELS.values()
    ),
)


@pytest.mark.parametrize(("scenario", "name", "n", "replicates"), PRIMARY_ROWS)
def test_every_primary_row_is_sized_for_its_bias(
    scenario: str, name: str, n: int, replicates: int
) -> None:
    bias, _ = projected(probe(SETS[scenario], scenario, name), n)
    assert bias_pass_probability(bias, replicates) >= PRIMARY_PASS


def test_law_i_at_40000_fails_the_rule() -> None:
    """The witness that law I's size was raised: at n = 40,000 its x2 contrast fails.

    The probe read +0.138 SD of bias on 600 samples there, so the primary bias rule passes with
    0.61 and the x0.5 calibration cell with 0.70.
    """
    contrast = probe("i@40000", "law_i", s.I_CONTRASTS["contrast_x2"])
    assert bias_pass_probability(projected(contrast, 40_000)[0], s.PRIMARY_REPLICATES) < 0.7
    other = probe("i@40000", "law_i", s.I_CONTRASTS["contrast_x0_5"])
    assert calibrated_pass_probability(other, 40_000, p.CALIBRATION_REPLICATES) < 0.8


@pytest.mark.parametrize("size", [size for size in p.RATE_SIZES if size != min(p.RATE_SIZES)])
@pytest.mark.parametrize("scenario", ["law_d_ltmle", "law_d_sdr"])
def test_every_positive_rung_is_sized_for_its_bias(scenario: str, size: int) -> None:
    row = probe("d@8000", scenario, s.D_CONTRASTS["static"])
    assert bias_pass_probability(projected(row, size)[0], p.RATE_REPLICATES) >= PRIMARY_PASS


def calibration_cells(replicates: int = p.CALIBRATION_REPLICATES) -> dict[str, float]:
    """Each calibration label's pass probability at its declared size and ``replicates``."""
    out = {}
    for label, (fit_set, scenario, name) in p.CALIBRATION.items():
        n = p.FIT_SETS[fit_set][2]
        out[label] = calibrated_pass_probability(
            probe(SETS[scenario], scenario, name), n, replicates
        )
    return out


def predictive_cells(replicates: int = p.CALIBRATION_REPLICATES) -> dict[str, float]:
    out = {}
    for label, (fit_set, scenario, name) in p.CALIBRATION.items():
        n = p.FIT_SETS[fit_set][2]
        out[label] = predictive_calibration(probe(SETS[scenario], scenario, name), n, replicates)
    return out


@pytest.mark.parametrize("label", list(p.CALIBRATION))
def test_every_calibration_cell_is_sized_for_its_pass_probability(label: str) -> None:
    assert calibration_cells()[label] >= CALIBRATION_PASS


def test_the_calibration_family_passes_jointly() -> None:
    """The independent product of the nine cells, with no credit for the D pairs."""
    assert float(np.prod(list(calibration_cells().values()))) >= FAMILY_PASS


@pytest.mark.parametrize("replicates", [3_600, 4_800])
def test_fewer_calibration_replications_fail_the_family_rule(replicates: int) -> None:
    """The witness that 6,000 is the smallest multiple of 1,200 the family rule admits."""
    assert float(np.prod(list(calibration_cells(replicates).values()))) < FAMILY_PASS


#: The probe set of each law at n = 2,000.
SMALL = {"law_d_ltmle": "d@2000", "law_s": "s@2000", "law_c": "c@2000", "law_m": "m@2000"}


@pytest.mark.parametrize("scenario", ["law_c", "law_m"])
def test_the_rule_binds_at_2000(scenario: str) -> None:
    """The witness: at n = 2,000 the calibration cell of laws C and M fails the rule.

    Both over-cover there (SE ratios 1.08 and 1.13).  The probe reads laws D and S as passing
    at n = 2,000, so for them the declared n = 8,000 rests on the shared size, not on this rule.
    """
    (label,) = [key for key, value in p.CALIBRATION.items() if value[1] == scenario]
    _, _, name = p.CALIBRATION[label]
    row = probe(SMALL[scenario], scenario, name)
    assert calibrated_pass_probability(row, 2_000, p.CALIBRATION_REPLICATES) < 0.01


def test_the_shrunken_se_controls_have_power() -> None:
    for label, (fit_set, scenario, name) in p.CALIBRATION.items():
        n = p.FIT_SETS[fit_set][2]
        bias, ratio = projected(probe(SETS[scenario], scenario, name), n)
        passed = calibrated_at(bias, ratio * p.SHRUNKEN_SE_FACTOR, p.CALIBRATION_REPLICATES)
        assert 1.0 - passed >= CONTROL_POWER, label


# ---------------------------------------------------------------------------- bands


@functools.cache
def accepted(replicates: int, accept: Callable[[Any], bool]) -> np.ndarray:
    """The success counts whose exact 99% interval satisfies ``accept``."""
    return np.array(
        [
            k
            for k in range(replicates + 1)
            if accept(clopper_pearson(k, replicates, confidence_level=MARGINS.confidence_level))
        ],
        dtype=int,
    )


def interval_pass_probability(rate: float, replicates: int, accept: Any) -> float:
    """The chance that the exact 99% interval of a binomial rate satisfies ``accept``."""
    rate = min(max(rate, 0.0), 1.0)
    return float(binom.pmf(accepted(replicates, accept), replicates, rate).sum())


def band_within(interval: Any) -> bool:
    return bool(interval.within(*MARGINS.calibration_coverage))


def control_below(interval: Any) -> bool:
    return bool(interval.high < 1.0 - MARGINS.alpha)


def band_reading(label: str, which: str) -> Any:
    _, scenario, _ = p.BANDS[label]
    return probe(SETS[scenario], scenario, "band", which)


@pytest.mark.parametrize("label", list(p.BANDS))
def test_every_band_label_is_sized_for_its_gate(label: str) -> None:
    """The band's joint coverage, moved one Monte Carlo SE either way, passes at 6,000."""
    band = band_reading(label, "band_covered")
    for coverage in (band.value - band.value_se, band.value + band.value_se):
        probability = interval_pass_probability(coverage, p.CALIBRATION_REPLICATES, band_within)
        assert probability >= CALIBRATION_PASS, (label, coverage, probability)


@pytest.mark.parametrize("label", list(p.BANDS))
def test_every_band_control_has_power(label: str) -> None:
    """The pointwise intervals, moved one Monte Carlo SE up, cover jointly below 0.95."""
    control = band_reading(label, "pointwise_covered")
    rate = control.value + control.value_se
    probability = interval_pass_probability(rate, p.CALIBRATION_REPLICATES, control_below)
    assert probability >= CONTROL_POWER, (label, probability)


# ---------------------------------------------------------------------- the increment


def increment_reading() -> tuple[Any, Any]:
    name = s.s_contrast(p.INCREMENT_HORIZON)
    return probe("s@8000", "law_s", name, "increment"), probe("s@8000", "law_s", name)


def increment_se(row: Any) -> float:
    """The paired bootstrap SE of the increment at the calibration count."""
    return float(row.value_se * np.sqrt(row.replicates / p.CALIBRATION_REPLICATES))


def increment_pass(value: float, se: float) -> float:
    """The chance the 99% upper endpoint of the increment is below the ceiling."""
    return float(norm.cdf((p.INCREMENT_CEILING - value) / se - Z99))


def test_the_increment_gate_is_sized() -> None:
    row, _ = increment_reading()
    se = increment_se(row)
    assert increment_pass(row.value, se) >= INCREMENT_PASS
    assert predictive(lambda value: increment_pass(value, se), (row.value, row.value_se)) >= (
        INCREMENT_PASS
    )


def test_the_increment_control_has_power() -> None:
    """The median SE times 1.05 adds 0.05 of the median SE ratio to the increment."""
    row, median = increment_reading()
    control = row.value + (p.INFLATED_SE_FACTOR - 1.0) * median.se_ratio
    assert 1.0 - increment_pass(control, increment_se(row)) >= CONTROL_POWER


# ---------------------------------------------------------------------- stability


#: The probe set of each stability law at its declared size.
STABILITY_SETS = {"d": "stability_d", "i": f"stability_i@{s.IPSI_N}"}


def stability_reading(label: str) -> Any:
    law, scenario, name = p.STABILITY[label]
    return probe(STABILITY_SETS[law], scenario, name, "stability")


def stability_pass(value: float, se: float) -> float:
    return float(norm.cdf((p.MAX_REPEAT_SPREAD_RATIO - value) / se - Z99))


@pytest.mark.parametrize("label", list(p.STABILITY))
def test_every_stability_label_has_split_spread_and_power(label: str) -> None:
    """A nonzero split spread, and a spread ratio whose 99% upper endpoint clears one."""
    row = stability_reading(label)
    assert row.share > 0
    se = float(row.value_se * np.sqrt(row.replicates / p.STABILITY_TRIALS))
    assert stability_pass(row.value, se) >= PRIMARY_PASS


def test_the_natural_course_has_no_split_spread() -> None:
    """Why ``ey_regimen[natural course]`` is not a stability name: no draw moves it."""
    name = f"ey_regimen[{s.I_REFERENCE}]"
    assert probe(STABILITY_SETS["i"], "law_i", name, "stability").share < 1e-12


# ---------------------------------------------------------------------- the declaration


def declaration() -> pd.DataFrame:
    """Every pass probability of the declaration, at the reading and predictively."""
    records: list[dict[str, Any]] = []
    for scenario, name, n, replicates in PRIMARY_ROWS:
        row = probe(SETS[scenario], scenario, name)
        records.append(
            {
                "rule": "primary bias",
                "cell": f"{scenario}/{name}",
                "pass": bias_pass_probability(projected(row, n)[0], replicates),
                "predictive": predictive_bias(row, n, replicates),
            }
        )
    point, forecast = calibration_cells(), predictive_cells()
    for label in p.CALIBRATION:
        records.append(
            {
                "rule": "calibration",
                "cell": label,
                "pass": point[label],
                "predictive": forecast[label],
            }
        )
    records.append(
        {
            "rule": "calibration family",
            "cell": "independent product",
            "pass": float(np.prod(list(point.values()))),
            "predictive": float(np.prod(list(forecast.values()))),
        }
    )
    for label in p.BANDS:
        for which, accept in (("band_covered", band_within), ("pointwise_covered", control_below)):
            row = band_reading(label, which)

            def at(rate: float, check: Any = accept) -> float:
                return interval_pass_probability(rate, p.CALIBRATION_REPLICATES, check)

            records.append(
                {
                    "rule": "band" if which == "band_covered" else "band control power",
                    "cell": label,
                    "pass": at(row.value),
                    "predictive": predictive(at, (row.value, row.value_se)),
                }
            )
    row, median = increment_reading()
    se = increment_se(row)
    control = row.value + (p.INFLATED_SE_FACTOR - 1.0) * median.se_ratio
    records.append(
        {
            "rule": "repeat_increment",
            "cell": f"h{p.INCREMENT_HORIZON}",
            "pass": increment_pass(row.value, se),
            "predictive": predictive(
                lambda value: increment_pass(value, se), (row.value, row.value_se)
            ),
        }
    )
    records.append(
        {
            "rule": "repeat_increment control power",
            "cell": f"h{p.INCREMENT_HORIZON}",
            "pass": 1.0 - increment_pass(control, se),
            "predictive": 1.0
            - predictive(lambda value: increment_pass(value, se), (control, row.value_se)),
        }
    )
    for label in p.STABILITY:
        row = stability_reading(label)
        se = float(row.value_se * np.sqrt(row.replicates / p.STABILITY_TRIALS))
        records.append(
            {
                "rule": "repeat_stability",
                "cell": label,
                "pass": stability_pass(row.value, se),
                "predictive": predictive(
                    lambda value, sd=se: stability_pass(value, sd), (row.value, row.value_se)
                ),
            }
        )
    return pd.DataFrame(records)


def test_the_declaration_states_every_probability() -> None:
    table = declaration()
    assert table[["pass", "predictive"]].notna().all().all()
    assert ((table["pass"] >= 0) & (table["pass"] <= 1 + 1e-12)).all()


if __name__ == "__main__":  # pragma: no cover - prints the declaration
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(declaration().round(4).to_string(index=False))
