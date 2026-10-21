"""The declared design of ``longitudinal-repeated-crossfit`` (``X46-plan.md`` 5.4).

Every number here was fixed before any registered run.  The readings come from the sizing
probe ``tests/diagnostics/x46_sizing/`` (``reading.csv``), which fits the study's own declared
repeated fits on seeds outside every declared stream.

* **Truths.**  Every truth is a finite sum on an exact law.  Law M's duration is centred at
  :data:`~tests.studies.longitudinal_repeated_crossfit.M_STAR`, which makes the exact intercept
  and slope curves uncorrelated; the slope is the uncentred slope and the intercept is
  :math:`b_0 + m^* b_1`.
* **Bounds.**  ``g_bounds = (0.01, 1)`` never binds at the truth on laws D, S, C and I, and the
  fallback-read bias bound of law I is below 0.1 standard errors at n = 80,000.
* **The sizing rule** (X19's projection, applied to X46's replication counts).  A pilot reading
  at ``n0`` projects its standardized bias as :math:`(|b| + m)\\sqrt{n_0 / n}` and its SE-ratio
  deficit as :math:`(1 - \\min(r, 1)) n_0 / n`.  Every primary row passes its bias rule with
  probability at least 0.95 at 1,600 replications.  Every calibration cell passes with
  probability at least 0.90 at 24,000, and the nine pass jointly, as the independent product,
  with probability at least 0.95.  3,600 and 4,800 replications fail the family rule, and
  24,000 is declared for a joint predictive pass near 0.89.
* **The joint predictive family.**  A bootstrap of the pilot samples draws the nine readings
  jointly, so the D pairs and the two law-I contrasts keep their correlation.  The nine
  cells then pass together with probability about 0.89 at 24,000 replications.  That is
  declared, with the red-cell reading of
  :data:`~tests.studies.longitudinal_repeated_crossfit_properties.RED_CELL_POLICY`; it is not
  a sizing rule, because its shortfall is pilot uncertainty that replications cannot remove.
* **The ladder.**  The 2,000 rung's control rule cannot fail at 800 replications.  The slope
  rows carry the check, and their pass probability is computed from the pilot's 2,000 and
  8,000 readings.
* **Law I's size.**  At n = 40,000 the probe read law I's contrasts at +0.138 and -0.124 SD of
  bias, so the primary rule passed with 0.61 and the family with 0.64.  A direct reading at
  80,000 (1,200 samples) passes every law-I cell, and the family with 0.98.
* **Controls.**  Every band label's pointwise control and the shrunken-SE controls fail with
  power at least 0.90.  The ``repeat_increment`` gate passes with probability at least 0.95,
  at the reading and predictively, and its inflated-SE control has power at least 0.90.
* **Predictive probabilities** (rule 7).  Each pass probability is also averaged over a normal
  draw of the reading about its value with its Monte Carlo SE.  :func:`declaration` prints both,
  and :data:`~tests.studies.longitudinal_repeated_crossfit_properties.PASS_PROBABILITIES`
  declares the predictive ones.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq
from scipy.stats import binom, chi2, norm
from scipy.stats import t as student

from tests import discrete_law_longitudinal as dlaw
from tests import discrete_law_longitudinal_incremental as ilaw
from tests.studies import longitudinal_incremental_common as common
from tests.studies import longitudinal_msm_properties as msm_law
from tests.studies import longitudinal_repeated_crossfit as s
from tests.studies import longitudinal_repeated_crossfit_properties as p
from tests.studies.canonical_full_refit_bootstrap import calibration_pass_probability
from tests.studies.evidence import clopper_pearson
from tests.studies.evidence.property_verdicts import (
    EXCLUDED_SLOPE,
    ROOT_N_SLOPE,
    ROOT_N_SLOPE_MARGIN,
)
from tests.studies.evidence.registry import ROOT, Margins, registered
from tests.studies.evidence.seeds import replicate_seed

pytestmark = pytest.mark.xdist_group("longitudinal_repeated_crossfit_design")

SIZING = ROOT / "tests" / "diagnostics" / "x46_sizing"
READING = pd.read_csv(SIZING / "reading.csv")
#: The pilot's per-sample readings: the median and draw-0 estimates and SEs of every name.
ROWS = pd.read_csv(SIZING / "rows.csv.gz")
#: The committed ``simultaneous_coverage`` and ``repeat_increment`` rows of the first declared
#: run (``properties.csv`` at ``PILOT_COMMIT``, 24,000 replications each).  That run read both
#: families off the calibration fits.  The rerun draws each family its own samples, so these
#: rows are an independent pilot for it.
PILOT = pd.read_csv(SIZING / "rerun_pilot.csv")
PILOT_COMMIT = "07092dd43980ea626c7fc435bd71e1ff9e8dcbfe"
#: The band arms of the pooled rule.  Law D's two bands read one sample and count once.
BAND_ARMS = ("d_ltmle", "i_ipsi", "s_ltmle", "c_ltmle", "m_ltmle")
BAND_ARM = {label: ("d_ltmle" if label == "d_sdr" else label) for label in p.BANDS}
#: The Monte Carlo draws of the pooled rule and their seed.
BAND_DRAWS = 200_000
BAND_SEED = 4646
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
#: The pilot bootstrap of the joint predictive family: draws and seed.
JOINT_DRAWS = 2_000
JOINT_SEED = 46
#: How far a declared probability may sit from the computed one: its two-decimal rounding,
#: plus the Monte Carlo error of the joint bootstrap.
DECLARED_TOLERANCE = 0.02


def test_the_declared_numbers() -> None:
    assert s.STUDY.publication_policy == "reporting"
    assert s.STUDY.reference is None
    assert s.STUDY.margins == MARGINS
    assert (s.PRIMARY_N, s.PRIMARY_REPLICATES, s.IPSI_N) == (8_000, 1_600, 80_000)
    assert (s.N_FOLDS, s.LEARNER_FOLDS, s.REPEATS, s.G_BOUNDS) == (5, 2, 3, (0.01, 1.0))
    assert (s.SEED, s.RESAMPLING_SEED) == (20261446, 2026144601)
    assert (p.CALIBRATION_N, p.CALIBRATION_REPLICATES) == (8_000, 24_000)
    assert (p.BAND_REPLICATES, p.INCREMENT_REPLICATES) == (2_400, 2_800)
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
    0.61 and the x0.5 calibration cell with 0.70 at the 6,000 replications of the first
    declaration.  At 24,000 that cell passes at n = 40,000 too, so the witness keeps 6,000.
    """
    contrast = probe("i@40000", "law_i", s.I_CONTRASTS["contrast_x2"])
    assert bias_pass_probability(projected(contrast, 40_000)[0], s.PRIMARY_REPLICATES) < 0.7
    other = probe("i@40000", "law_i", s.I_CONTRASTS["contrast_x0_5"])
    assert calibrated_pass_probability(other, 40_000, 6_000) < 0.8


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


def calibrated_many(bias: np.ndarray, ratio: np.ndarray, replicates: int) -> np.ndarray:
    """:func:`calibrated_at` over arrays: the same normal approximation, vectorized.

    ``test_the_vectorized_calibration_rule_is_the_scalar_one`` pins it to the scalar rule.
    """
    z = Z99
    coverage = norm.cdf(norm.ppf(0.975) * ratio - bias) - norm.cdf(-norm.ppf(0.975) * ratio - bias)
    spread = ratio * np.sqrt(1.0 / (2.0 * (replicates - 1)))
    low, high = MARGINS.calibration_se_ratio
    inside = norm.cdf((high - z * spread - ratio) / spread) - norm.cdf(
        (low + z * spread - ratio) / spread
    )
    width = np.sqrt(coverage * (1.0 - coverage) / replicates)
    floor, ceiling = MARGINS.calibration_coverage
    covered = norm.cdf((ceiling - z * width - coverage) / width) - norm.cdf(
        (floor + z * width - coverage) / width
    )
    return np.clip(inside, 0.0, None) * np.clip(covered, 0.0, None)


def test_the_vectorized_calibration_rule_is_the_scalar_one() -> None:
    bias = np.array([0.0, 0.03, 0.08, 0.2])
    ratio = np.array([1.0, 1.02, 0.97, 1.06])
    for replicates in (3_600, 24_000):
        scalar = [calibrated_at(b, r, replicates) for b, r in zip(bias, ratio, strict=True)]
        np.testing.assert_allclose(calibrated_many(bias, ratio, replicates), scalar, atol=1e-12)


@functools.cache
def joint_predictive_family(replicates: int = p.CALIBRATION_REPLICATES) -> float:
    """The chance the nine calibration cells pass together, with correlated pilot errors.

    Each probe set's samples are resampled jointly, so labels read off one sample keep their
    correlation: D-``LTMLE`` with D-``SDR`` (1.000 on the SE ratio) and the two law-I contrasts
    (0.595).  Each draw gives every label its standardized bias and SE ratio, and the cells'
    pass probabilities multiply within the draw.  The sets are independent seed streams, so
    the family is the product of the sets' mean products.  The readings are at the declared
    sizes, so nothing is projected.
    """
    rng = np.random.default_rng(JOINT_SEED)
    by_set: dict[str, list[tuple[str, str]]] = {}
    for _, scenario, name in p.CALIBRATION.values():
        by_set.setdefault(SETS[scenario], []).append((scenario, name))
    family = 1.0
    for cell_set, members in by_set.items():
        sample = ROWS.loc[ROWS["set"] == cell_set]
        order = np.sort(sample["replicate"].unique())
        index = rng.integers(0, len(order), size=(JOINT_DRAWS, len(order)))
        product = np.ones(JOINT_DRAWS)
        for scenario, name in members:
            rows = sample.loc[(sample["scenario"] == scenario) & (sample["name"] == name)]
            rows = rows.set_index("replicate").loc[order]
            errors = (rows["psi_med"] - rows["truth"]).to_numpy()[index]
            spread = errors.std(axis=1, ddof=1)
            bias = np.abs(errors.mean(axis=1)) / spread
            ratio = rows["se_med"].to_numpy()[index].mean(axis=1) / spread
            product *= calibrated_many(bias, ratio, replicates)
        family *= float(product.mean())
    return family


def test_the_joint_predictive_family_is_declared() -> None:
    """About 0.89 at 24,000: declared beside the red-cell reading, not used to size the study.

    The correlation credit is the witness that the draws are joint: the independent product
    of the nine predictive cells is lower.
    """
    joint = joint_predictive_family()
    declared = p.PASS_PROBABILITIES["interval_calibration/calibration_family"]
    assert abs(joint - declared) <= DECLARED_TOLERANCE
    assert joint > float(np.prod(list(predictive_cells().values())))
    assert joint < FAMILY_PASS
    assert "0.89 predictively" in " ".join(p.RED_CELL_POLICY)


@pytest.mark.parametrize("replicates", [3_600, 4_800])
def test_fewer_calibration_replications_fail_the_family_rule(replicates: int) -> None:
    """The witness that 3,600 and 4,800 replications fail the family rule."""
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


def pilot(family: str, cell: str) -> Any:
    """One row of the rerun pilot: the committed rows of the first declared run."""
    (row,) = PILOT[(PILOT["family"] == family) & (PILOT["cell"] == cell)].itertuples()
    return row


@functools.cache
def accepted_range(replicates: int, band: bool) -> tuple[int, int]:
    """The smallest and largest success count whose exact 99% interval passes the band rule.

    ``band`` reads the joint-coverage band rule, and its absence reads the pointwise control
    rule (the upper endpoint below the nominal rate).  The accepted counts are contiguous,
    which the assertion checks, so a binomial CDF difference gives the pass probability.
    """
    accept = band_within if band else control_below
    counts = accepted(replicates, accept)
    assert counts.max() - counts.min() + 1 == len(counts)
    return int(counts.min()), int(counts.max())


def band_within(interval: Any) -> bool:
    return bool(interval.within(*MARGINS.calibration_coverage))


def control_below(interval: Any) -> bool:
    return bool(interval.high < 1.0 - MARGINS.alpha)


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


def interval_pass_probability(rate: Any, replicates: int, band: bool) -> Any:
    """The chance that the exact 99% interval of a binomial rate passes the band or control rule."""
    low, high = accepted_range(replicates, band)
    rate = np.clip(rate, 0.0, 1.0)
    return binom.cdf(high, replicates, rate) - binom.cdf(low - 1, replicates, rate)


def pilot_se(row: Any) -> float:
    """The binomial standard error of a pilot rate."""
    return float(np.sqrt(row.value * (1.0 - row.value) / row.replicates))


def pooled_model() -> tuple[float, float, float]:
    """The pooled mean of the five band arms, its variance, and an upper bound of tau.

    The arm-pooled predictive rule: arm ``j``'s true joint coverage is
    ``m + e + u_j`` with ``e ~ N(0, v)`` shared and ``u_j ~ N(0, tau^2)``.  ``m`` is the
    precision-weighted mean of the pilot rates at ``tau``, ``v`` is the variance of that mean
    (the arms' sampling variance over their effective number), and ``tau`` is the upper bound
    of the Q-profile interval at the 10th percentile of the chi-square law of ``Q``, a
    one-sided 90% bound.  The shared pilot level is 0.80; this bound is a deliberate, more
    conservative departure, so ``tau`` and the replication count are not smaller than 0.80
    gives.  The two law-D arms read one sample and move together, so they count once, as law D.
    The family product therefore leaves the ``d_sdr`` gate out; its pilot rate agrees with the
    ``d_ltmle`` rate to 4e-5.
    """
    rates = np.array(
        [pilot("simultaneous_coverage", f"{a}__simultaneous_band").value for a in BAND_ARMS]
    )
    variance = np.array(
        [
            pilot_se(pilot("simultaneous_coverage", f"{a}__simultaneous_band")) ** 2
            for a in BAND_ARMS
        ]
    )

    def profile(tau: float) -> tuple[float, float, float]:
        weight = 1.0 / (variance + tau**2)
        mean = float(np.sum(weight * rates) / np.sum(weight))
        return float(np.sum(weight * (rates - mean) ** 2)), mean, float(1.0 / np.sum(weight))

    target = float(chi2.ppf(0.10, len(rates) - 1))
    tau = 0.0 if profile(0.0)[0] <= target else brentq(lambda t: profile(t)[0] - target, 0.0, 1.0)
    _, mean, spread = profile(tau)
    return mean, spread, float(tau)


@functools.cache
def band_draws() -> tuple[np.ndarray, np.ndarray]:
    """The predictive draws of the five band arms' true rates and of their five controls'."""
    mean, spread, tau = pooled_model()
    rng = np.random.default_rng(BAND_SEED)
    shared = rng.normal(0.0, np.sqrt(spread), (BAND_DRAWS, 1))
    arms = mean + shared + rng.normal(0.0, tau, (BAND_DRAWS, len(BAND_ARMS)))
    controls = np.column_stack(
        [
            rng.normal(row.value, pilot_se(row), BAND_DRAWS)
            for row in (
                pilot("simultaneous_coverage", f"{a}__pointwise_joint_control") for a in BAND_ARMS
            )
        ]
    )
    return arms, controls


def band_passes(replicates: int) -> tuple[np.ndarray, np.ndarray]:
    """Each draw's pass probability for the five bands and the five controls."""
    arms, controls = band_draws()
    return (
        interval_pass_probability(arms, replicates, True),
        interval_pass_probability(controls, replicates, False),
    )


def band_family(replicates: int = p.BAND_REPLICATES) -> float:
    """The chance that every band passes and every pointwise control has power together."""
    bands, controls = band_passes(replicates)
    return float(np.mean(np.prod(bands, axis=1) * np.prod(controls, axis=1)))


def test_the_band_pilot_is_the_first_declared_run() -> None:
    """The pilot rows are the committed ``properties.csv`` at the artifact commit, 24,000 each."""
    assert set(PILOT["artifact_commit"]) == {PILOT_COMMIT}
    assert set(PILOT["replicates"]) == {24_000}
    assert len(PILOT) == 2 + 2 * len(p.BANDS)


def test_the_band_family_passes_jointly_by_the_pooled_rule() -> None:
    """2,400 is the smallest multiple of 400 whose family-wise predictive pass is at least 0.95."""
    assert band_family(p.BAND_REPLICATES) >= FAMILY_PASS
    assert band_family(p.BAND_REPLICATES - 400) < FAMILY_PASS


def test_the_pooled_model_is_read_from_the_pilot() -> None:
    mean, spread, tau = pooled_model()
    assert 0.950 < mean < 0.953
    assert 0.001 < np.sqrt(spread) < 0.003
    assert 0.0 < tau < 0.01


@pytest.mark.parametrize("label", list(p.BANDS))
def test_every_band_label_is_sized_for_its_gate(label: str) -> None:
    """Each band passes at the pilot rate, and each pooled arm clears the cell floor."""
    row = pilot("simultaneous_coverage", f"{label}__simultaneous_band")
    assert float(interval_pass_probability(row.value, p.BAND_REPLICATES, True)) >= CALIBRATION_PASS
    arm = band_passes(p.BAND_REPLICATES)[0][:, BAND_ARMS.index(BAND_ARM[label])]
    assert float(arm.mean()) >= CALIBRATION_PASS


@pytest.mark.parametrize("label", list(p.BANDS))
def test_every_band_control_has_power(label: str) -> None:
    """The pointwise intervals, moved one Monte Carlo SE up, cover jointly below 0.95."""
    row = pilot("simultaneous_coverage", f"{label}__pointwise_joint_control")
    rate = row.value + pilot_se(row)
    assert float(interval_pass_probability(rate, p.BAND_REPLICATES, False)) >= CONTROL_POWER


# ---------------------------------------------------------------------- the increment


def increment_pilot_se(row: Any) -> float:
    """The bootstrap SE of a pilot increment at 24,000: its 99% interval over twice Z99."""
    return float((row.ci_upper - row.ci_lower) / (2.0 * Z99))


def increment_se(row: Any, replicates: int = p.INCREMENT_REPLICATES) -> float:
    """The paired bootstrap SE of the increment at ``replicates``, from the pilot's."""
    return increment_pilot_se(row) * float(np.sqrt(row.replicates / replicates))


def increment_pass(value: float, se: float) -> float:
    """The chance the 99% upper endpoint of the increment is below the ceiling."""
    return float(norm.cdf((p.INCREMENT_CEILING - value) / se - Z99))


def increment_probabilities(replicates: int = p.INCREMENT_REPLICATES) -> tuple[float, float]:
    """The predictive chance that the gate passes and that the inflated-SE control has power.

    One arm has no between-arm term, so the rule reduces to the pilot's own uncertainty: the
    true increment is a normal draw about the pilot reading with the pilot's bootstrap SE.
    """
    gate, control = (
        pilot("repeat_increment", "h3__repeat_median"),
        pilot("repeat_increment", "h3__inflated_se_control"),
    )
    passed = predictive(
        lambda value: increment_pass(value, increment_se(gate, replicates)),
        (gate.value, increment_pilot_se(gate)),
    )
    power = 1.0 - predictive(
        lambda value: increment_pass(value, increment_se(control, replicates)),
        (control.value, increment_pilot_se(control)),
    )
    return passed, power


def test_the_increment_gate_is_sized() -> None:
    """2,800 is the smallest multiple of 400 whose predictive gate pass is at least 0.95."""
    gate = pilot("repeat_increment", "h3__repeat_median")
    assert increment_pass(gate.value, increment_se(gate)) >= INCREMENT_PASS
    assert increment_probabilities(p.INCREMENT_REPLICATES)[0] >= INCREMENT_PASS
    assert increment_probabilities(p.INCREMENT_REPLICATES - 400)[0] < INCREMENT_PASS


def test_the_increment_control_has_power() -> None:
    """The median SE times 1.05 adds 0.05 of the median SE ratio to the increment."""
    assert increment_probabilities(p.INCREMENT_REPLICATES)[1] >= CONTROL_POWER


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
    assert row.share > p.STABILITY_SPREAD_FLOOR
    assert one_draw_share(*p.STABILITY[label]) > p.STABILITY_SPREAD_FLOOR
    se = float(row.value_se * np.sqrt(row.replicates / p.STABILITY_TRIALS))
    assert stability_pass(row.value, se) >= PRIMARY_PASS


def one_draw_share(law: str, scenario: str, name: str) -> float:
    """The pilot's one-draw split variance over the mean reported variance on a fixed sample.

    The quantity ``stability_verdicts`` checks against ``STABILITY_SPREAD_FLOOR`` at run time.
    The reported SEs come from the calibration pilot at the same law and size.
    """
    trials = pd.read_csv(SIZING / "stability.csv.gz")
    one = trials.loc[
        (trials["set"] == STABILITY_SETS[law])
        & (trials["scenario"] == scenario)
        & (trials["name"] == name),
        "one",
    ]
    errors = ROWS.loc[
        (ROWS["set"] == SETS[scenario]) & (ROWS["scenario"] == scenario) & (ROWS["name"] == name),
        "se_med",
    ]
    return float(one.var(ddof=1) / np.mean(errors**2))


def test_the_natural_course_has_no_split_spread() -> None:
    """Why ``ey_regimen[natural course]`` is not a stability name: no draw moves it.

    It is also the witness that the floor is not vacuous: both of its readings, the median
    displacement share (1.7e-15) and the one-draw split variance (4.1e-26), fall below it.
    """
    name = f"ey_regimen[{s.I_REFERENCE}]"
    assert probe(STABILITY_SETS["i"], "law_i", name, "stability").share < p.STABILITY_SPREAD_FLOOR
    assert one_draw_share("i", "law_i", name) < p.STABILITY_SPREAD_FLOOR


# ---------------------------------------------------------------------------- the ladder


def test_the_smallest_rung_cannot_fail() -> None:
    """At 800 replications the control rule's two regions cover every count.

    A rung passes when its exact 99% upper bound is below 0.95 or its lower bound is at least
    0.90.  Every interval at 800 is narrower than the 0.05 between them, so no count fails.
    The rung is therefore not a slow-convergence control (``RED_CELL_POLICY`` rule 8).
    """

    def resolves(interval: Any) -> bool:
        return bool(interval.high < 1.0 - MARGINS.alpha or interval.low >= MARGINS.coverage_floor)

    assert len(accepted(p.RATE_REPLICATES, resolves)) == p.RATE_REPLICATES + 1
    assert "cannot fail" in " ".join(p.RED_CELL_POLICY)


def rung_reading(scenario: str, cell_set: str, column: str) -> tuple[float, float, int]:
    """``(log statistic, its variance per replicate, samples)`` of one pilot rung.

    ``psi_med`` reads the log empirical SD of the error, whose variance is about
    ``(kurtosis - 1) / (4 R)``; ``se_med`` reads the log mean reported SE, whose variance is
    about ``cv**2 / R``.
    """
    name = s.D_CONTRASTS["static"]
    rows = ROWS.loc[(ROWS["set"] == cell_set) & (ROWS["scenario"] == scenario)]
    rows = rows.loc[rows["name"] == name]
    if column == "psi_med":
        errors = (rows["psi_med"] - rows["truth"]).to_numpy()
        centred = errors - errors.mean()
        kurtosis = float(np.mean(centred**4) / np.mean(centred**2) ** 2)
        return float(np.log(errors.std(ddof=1))), (kurtosis - 1.0) / 4.0, len(rows)
    values = rows["se_med"].to_numpy()
    return float(np.log(values.mean())), float(values.var(ddof=1) / values.mean() ** 2), len(rows)


def projected_slope(at_2000: float, at_8000: float) -> float:
    """The three-rung OLS slope, with the 32,000 rung projected from the 2,000 and 8,000 rungs.

    The statistic times root n is ``a (1 + b / sqrt(n))``: the excess over its limit decays as
    ``n**-0.5``.  That is the slowest of the decays the two rungs admit (an excess decaying as
    ``1/n``, or none past 8,000, projects a slope closer to -0.5), so it is the conservative
    reading.
    """
    scaled = np.exp(np.array([at_2000, at_8000]) + 0.5 * np.log([2_000.0, 8_000.0]))
    ratio = scaled[0] / scaled[1]
    b = (ratio - 1.0) / (1.0 / np.sqrt(2_000.0) - ratio / np.sqrt(8_000.0))
    a = scaled[1] / (1.0 + b / np.sqrt(8_000.0))
    at_32000 = np.log(a * (1.0 + b / np.sqrt(32_000.0))) - 0.5 * np.log(32_000.0)
    return float((at_32000 - at_2000) / np.log(32_000.0 / 2_000.0))


def slope_pass(slope: float, se: float) -> float:
    """The chance the slope's 99% interval lies within 0.125 of -0.5 and excludes -0.25."""
    half = Z99 * se
    low, high = ROOT_N_SLOPE - ROOT_N_SLOPE_MARGIN, ROOT_N_SLOPE + ROOT_N_SLOPE_MARGIN
    inside = norm.cdf((min(high, EXCLUDED_SLOPE) - half - slope) / se) - norm.cdf(
        (low + half - slope) / se
    )
    return float(max(inside, 0.0))


def ladder_probabilities(scenario: str, column: str) -> tuple[float, float]:
    """The slope row's pass probability at the pilot reading and predictively."""
    log_2000, unit_2000, pilot_2000 = rung_reading(scenario, "d@2000", column)
    log_8000, unit_8000, pilot_8000 = rung_reading(scenario, "d@8000", column)
    # The fitted slope of three equally spaced log sizes reads only the end rungs; the
    # 32,000 rung takes the 8,000 rung's unit variance.
    se = float(np.sqrt((unit_2000 + unit_8000) / p.RATE_REPLICATES) / np.log(32_000.0 / 2_000.0))

    def at(first: float, second: float) -> float:
        return slope_pass(projected_slope(first, second), se)

    return at(log_2000, log_8000), predictive(
        at,
        (log_2000, float(np.sqrt(unit_2000 / pilot_2000))),
        (log_8000, float(np.sqrt(unit_8000 / pilot_8000))),
    )


LADDER = {
    f"{label}__{suffix}": (scenario, column)
    for label, scenario in zip(s.RATE_LABELS, ("law_d_ltmle", "law_d_sdr"), strict=True)
    for suffix, column in (("empirical_sd", "psi_med"), ("reported_se", "se_med"))
}


@pytest.mark.parametrize("cell", list(LADDER))
def test_every_slope_row_is_sized(cell: str) -> None:
    """The pilot's slope projects near -0.57 for the spread; each row passes at the reading."""
    reading, forecast = ladder_probabilities(*LADDER[cell])
    assert reading >= PRIMARY_PASS
    assert abs(forecast - p.PASS_PROBABILITIES[f"root_n_rate/{cell}"]) <= DECLARED_TOLERANCE


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
            "cell": "independent product (pass); joint bootstrap (predictive)",
            "pass": float(np.prod(list(point.values()))),
            "predictive": joint_predictive_family(),
        }
    )
    bands, controls = band_passes(p.BAND_REPLICATES)
    for label in p.BANDS:
        arm = BAND_ARMS.index(BAND_ARM[label])
        for which, forecast, band in (
            ("band", bands, True),
            ("band control power", controls, False),
        ):
            kind = "simultaneous_band" if band else "pointwise_joint_control"
            row = pilot("simultaneous_coverage", f"{label}__{kind}")
            records.append(
                {
                    "rule": which,
                    "cell": label,
                    "pass": float(interval_pass_probability(row.value, p.BAND_REPLICATES, band)),
                    "predictive": float(forecast[:, arm].mean()),
                }
            )
    records.append(
        {
            "rule": "band family",
            "cell": "band_family",
            "pass": float(
                np.prod(
                    [
                        interval_pass_probability(
                            pilot("simultaneous_coverage", f"{label}__simultaneous_band").value,
                            p.BAND_REPLICATES,
                            True,
                        )
                        for label in BAND_ARMS
                    ]
                )
            ),
            "predictive": band_family(),
        }
    )
    passed, power = increment_probabilities()
    gate = pilot("repeat_increment", "h3__repeat_median")
    records.append(
        {
            "rule": "repeat_increment",
            "cell": f"h{p.INCREMENT_HORIZON}",
            "pass": increment_pass(gate.value, increment_se(gate)),
            "predictive": passed,
        }
    )
    control = pilot("repeat_increment", "h3__inflated_se_control")
    records.append(
        {
            "rule": "repeat_increment control power",
            "cell": f"h{p.INCREMENT_HORIZON}",
            "pass": 1.0 - increment_pass(control.value, increment_se(control)),
            "predictive": power,
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
    for cell, (scenario, column) in LADDER.items():
        reading, ahead = ladder_probabilities(scenario, column)
        records.append({"rule": "root_n_rate", "cell": cell, "pass": reading, "predictive": ahead})
    return pd.DataFrame(records)


def test_the_declaration_states_every_probability() -> None:
    table = declaration()
    assert table[["pass", "predictive"]].notna().all().all()
    assert ((table["pass"] >= 0) & (table["pass"] <= 1 + 1e-12)).all()


def declared_key(rule: str, cell: str) -> str | None:
    """The ``PASS_PROBABILITIES`` key of one declaration row, if the row is declared there."""
    family = {
        "calibration": "interval_calibration",
        "band": "simultaneous_coverage",
        "band family": "simultaneous_coverage",
        "repeat_increment": "repeat_increment",
        "repeat_stability": "repeat_stability",
        "root_n_rate": "root_n_rate",
    }
    if rule == "calibration family":
        return "interval_calibration/calibration_family"
    if rule == "primary bias" and cell.startswith("law_i/"):
        names = {name: label for label, name in s.IPSI_PRIMARY_LABELS.items()}
        return f"ipsi_primary/{names[cell.split('/', 1)[1]]}"
    return f"{family[rule]}/{cell}" if rule in family else None


def test_every_declared_pass_probability_is_the_computed_one() -> None:
    """``PASS_PROBABILITIES`` states each gated cell's predictive probability, to 0.02."""
    table = declaration()
    computed = {
        key: float(row.predictive)
        for row in table.itertuples()
        if (key := declared_key(str(row.rule), str(row.cell))) is not None
    }
    assert set(computed) == set(p.PASS_PROBABILITIES)
    for key, value in computed.items():
        assert abs(value - p.PASS_PROBABILITIES[key]) <= DECLARED_TOLERANCE, (key, value)


if __name__ == "__main__":  # pragma: no cover - prints the declaration
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(declaration().round(4).to_string(index=False))
