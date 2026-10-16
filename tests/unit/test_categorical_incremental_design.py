"""The declared design of the study ``categorical-incremental`` (X43, ``X43-plan.md`` 5).

Every number here is fixed before any registered run.

* Every truth is a finite sum on an exact law, checked two ways to 1e-12.
* Every efficiency bound is the closed-form influence function of ``X43-plan.md`` 2.1, which
  agrees with the complex-step Gateaux derivative of the functional to 1e-10.
* Law ``P``'s gated contrast carries a mechanism share of at least 0.35 (plan rev 2,
  finding 6), so the curve without the mechanism term under-covers (limit below 0.90).
* The controls' limits: the dropped-term and binary-blip curves lie below the calibration
  band; the complete-case fit of law ``L`` misses its truth by more than 2 of its standard
  errors with probability at least 0.90 per fit (finding 2).
* The power cell has exact-law power in [0.80, 0.90].
* The sizing rules, from the family P pilot of 400 draws per rung
  (``cleverly-beta-plans/reviews/x43/design-pilot/``, ``pilot_p.py`` and ``summarize_p.py``):
  the primary size is the smallest rung at which every primary contrast has a standardized
  bias of at most 0.25, and each calibration cell's replicate count follows the arm-pooled
  predictive rule (the smallest multiple of 400 with family-wise pass probability at least
  0.95), read from an 8,400-draw calibration pilot at n = 4,000 (``pilot_cal_p.py``).
* The softmax witness (finding 1): the median movement of the targeted mechanism in the
  cross-fitted point cells is nonzero.

Family L's pilot waits for the longitudinal layer (``X43-plan.md`` section 7, stage 2); its
sizes are provisional, and the tests that read it are added with it.

Probes before declaration read seeds from ``SeedSequence([4343, ...])``; no declared stream
reads them.  The study's two seeds are new, and no registered study uses either.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom, chi2, norm
from scipy.stats import t as student

import cleverly.fluctuation.mechanism as mechanism_module
from cleverly._support import classify_key
from tests import discrete_law_categorical_ipsi_longitudinal as long_law
from tests import discrete_law_categorical_ipsi_point as point
from tests.studies import canonical_categorical_incremental as study
from tests.studies import categorical_incremental_common as common
from tests.studies import categorical_incremental_properties as properties
from tests.studies.canonical_full_refit_bootstrap import calibration_pass_probability
from tests.studies.evidence.inference import clopper_pearson
from tests.studies.evidence.property_verdicts import design_power, power_cell_pass_probability
from tests.studies.evidence.registry import Margins, registered
from tests.studies.intervention_study_helpers import incremental_estimates
from tests.studies.support_cases import CASES
from tests.unit._source_mutation import mutate

pytestmark = pytest.mark.xdist_group("categorical_incremental_design")

PRIMARY_PASS = 0.95
FAMILY_FLOOR = 0.95
CELL_FLOOR = 0.90
REPLICATE_GRAIN = 400
TAU_LEVEL = 0.80
#: The standardized-bias ceiling of the primary sizing rule (``X43-plan.md`` 5.3).
SIZING_BIAS = 0.25


def test_the_declared_numbers() -> None:
    assert study.STUDY.publication_policy == "gated"
    assert study.STUDY.margins == Margins()
    assert study.STUDY.reference is None and study.STUDY.pending_reference is None
    assert (study.PRIMARY_REPLICATES, study.FAILURE_BOUND) == (1_000, 0.01)
    assert properties.FAILURE_BOUND == 0.01
    assert study.SEED == 20261843 and study.RESAMPLING_SEED == 2026184301
    assert properties.EFFICIENCY_RATIO_BAND == (0.90, 1.10)
    assert (properties.NULL_N, properties.NULL_REPLICATES) == (4_000, 2_000)
    assert properties.CALIBRATION_REPLICATES % REPLICATE_GRAIN == 0
    assert study.PRIMARY_N >= study.POINT_PILOT_N
    assert {folds for _, _, folds in study.SCENARIOS.values()} == {1, common.N_FOLDS}


def test_the_study_draws_from_its_own_seeds() -> None:
    mine = {study.SEED, study.RESAMPLING_SEED}
    others = {
        seed
        for record in registered()
        if record.slug != study.STUDY.slug
        for seed in (record.seed, record.resampling_seed)
    }
    assert not mine & others


# ------------------------------------------------------------------------------- truths


@pytest.mark.parametrize("law", list(point.LAWS.values()), ids=list(point.LAWS))
def test_the_natural_course_is_the_observed_mean(law: Any) -> None:
    """A second route to a truth: the factor 1 leaves the law's own mean."""
    observed = float((law.masses * law.mu).sum())
    assert point.truth(law, f"ey_ipsi[{point.REFERENCE}]") == pytest.approx(observed, abs=1e-12)


def test_the_point_truths_by_a_second_route() -> None:
    """The tilted mean longhand, cell by cell, against the vectorised functional."""
    for name, tilt in point.TILTS.items():
        f = point.factor_vector(tilt)
        total = 0.0
        for w in range(len(point.P_W)):
            weights = [f[a] * point.G[w, a] for a in range(point.K)]
            total += (
                point.P_W[w]
                * sum(weight * point.MU[w, a] for a, weight in enumerate(weights))
                / sum(weights)
            )
        assert point.truth(point.LAW_P, f"ey_ipsi[{name}]") == pytest.approx(total, abs=1e-12)
    assert study.TRUTH["point_crossfit"]["ate_ipsi[T_a vs natural course]"] == pytest.approx(
        -0.8123799094105730, abs=1e-12
    )


def test_the_stratum_truths_average_to_the_marginal_truth() -> None:
    shares = [
        float(point.P_W[[w for w, v in enumerate(point.STRATA_OF_W) if v == s]].sum())
        for s in (0, 1)
    ]
    parts = [point.truth(point.stratum_law(point.LAW_P, s), point.GATED) for s in (0, 1)]
    assert float(np.dot(shares, parts)) == pytest.approx(
        point.truth(point.LAW_P, point.GATED), abs=1e-12
    )


def test_the_sharp_null_has_one_mean() -> None:
    values = {round(point.truth(point.LAW_P0, name), 14) for name in point.MEANS}
    assert len(values) == 1
    assert abs(properties.NULL_TRUTH) < 1e-12


def test_dropout_at_random_leaves_every_plan_value() -> None:
    """A nonzero witness of the censoring axes: law ``L`` has the targets of ``L0``."""
    for plan in long_law.PLANS.values():
        assert long_law.truth(long_law.LAW_L, plan) == pytest.approx(
            long_law.truth(long_law.LAW_L0, plan), abs=1e-12
        )
    assert long_law.LAW_L.probs.shape != long_law.LAW_L0.probs.shape
    observed = float((long_law.LAW_L0.probs * long_law.LAW_L0.mu).sum())
    natural = long_law.PLANS[long_law.REFERENCE]
    assert long_law.truth(long_law.LAW_L0, natural) == pytest.approx(observed, abs=1e-12)


def test_the_longitudinal_dropout_lies_in_its_declared_range() -> None:
    target = long_law.LAW_L
    k = len(target.axes)
    probs = target.probs
    for t in (1, 2):
        c = target.axis_of("censoring", t)
        upper = probs.sum(axis=tuple(range(c + 1, k)))
        lower = upper.sum(axis=c)
        kept = np.take(upper, 1, axis=c)
        reached = lower > 0
        # Only histories still at risk: after a dropout the later axis is a structural zero.
        omega = kept[reached] / lower[reached]
        # A history that dropped out at the first axis has no second decision: kept 0 or 1.
        omega = omega[(omega > 0.0) & (omega < 1.0)]
        assert omega.min() >= 0.85 - 1e-12 and omega.max() <= 0.97 + 1e-12
        assert omega.max() - omega.min() > 0.05


def test_the_longitudinal_mechanisms_keep_every_level() -> None:
    """Every node's propensity of every level lies in [0.07, 0.78] at every history.

    The plan's [0.08, 0.70] rule is law ``P``'s.  Law ``L``'s propensities span 0.070 to
    0.774 at the second node, above ``G_BOUNDS`` at every history.
    """
    for axis in ("A1", "A2"):
        target = long_law.LAW_L0
        j = [a.name for a in target.axes].index(axis)
        k = len(target.axes)
        upper = target.probs.sum(axis=tuple(range(j + 1, k)))
        share = upper / upper.sum(axis=j, keepdims=True)
        assert share.min() >= 0.07 and share.max() <= 0.78
        assert share.min() > common.G_BOUNDS[0]


# ------------------------------------------------------------------ efficiency bounds


@pytest.mark.parametrize("name", point.MEANS + point.CONTRASTS)
def test_the_closed_form_curve_is_the_gateaux_derivative(name: str) -> None:
    parts = point.eif_parts(point.LAW_P, name)
    for w in range(len(point.P_W)):
        for a in range(point.K):
            assert point.gateaux(point.LAW_P, name, w, a) == pytest.approx(
                parts.mechanism[w, a] + parts.baseline[w], abs=1e-10
            )


def test_the_natural_course_bound_is_the_outcome_sd() -> None:
    """The factor 1 makes the curve ``Y - E(Y)``, so its SD is the outcome's."""
    law = point.LAW_P
    mean = float((law.masses * law.mu).sum())
    variance = float((law.masses * (law.mu - mean) ** 2).sum()) + point.SIGMA**2
    assert point.eif_sd(law, f"ey_ipsi[{point.REFERENCE}]") == pytest.approx(
        math.sqrt(variance), abs=1e-12
    )
    target = long_law.LAW_L0
    mean = float((target.probs * target.mu).sum())
    variance = float((target.probs * (target.mu - mean) ** 2).sum()) + 1.0
    natural = long_law.PLANS[long_law.REFERENCE]
    assert long_law.eif_sd(target, natural) == pytest.approx(math.sqrt(variance), abs=1e-10)


def test_the_declared_bounds() -> None:
    expected = {
        "point_contrast_t_a": 0.946801,
        "point_contrast_t_b": 1.005315,
        "point_known_t_a": 0.580967,
    }
    for label, value in expected.items():
        assert properties.EFFICIENCY_SD[label] == pytest.approx(value, abs=1e-6), label
    # The known-mechanism bound is the curve without the mechanism term.
    share = point.mechanism_share(point.LAW_P, point.GATED)
    assert properties.EFFICIENCY_SD["point_known_t_a"] ** 2 == pytest.approx(
        (1.0 - share) * properties.EFFICIENCY_SD["point_contrast_t_a"] ** 2, rel=1e-12
    )


# ---------------------------------------------------------------------- the witnesses


def test_the_mechanism_term_carries_its_share() -> None:
    """Finding 6: the gated contrast's mechanism share is at least 0.35, every level scores."""
    assert point.mechanism_share(point.LAW_P, point.GATED) >= 0.35
    assert point.mechanism_share(point.LAW_P, properties.CONTRAST_T_B) >= 0.35
    for tilt in (point.T_A, point.T_B):
        coefficient = np.abs(point.mechanism_coefficients(point.LAW_P, tilt))
        assert coefficient.max(axis=0).min() >= 0.02
    assert point.G.min() >= 0.08 and point.G.max() <= 0.70


def _coverage_limit(ratio: float) -> float:
    return float(2.0 * norm.cdf(norm.ppf(0.975) * ratio) - 1.0)


def _below_band_probability(ratio: float, replicates: int, noise: float = 0.0) -> float:
    """The chance the 99% SE-ratio interval lies wholly below the calibration band."""
    z = float(norm.ppf(0.995))
    spread = ratio * math.sqrt(1.0 / (2.0 * (replicates - 1)) + noise**2 / replicates)
    return float(norm.cdf((Margins().calibration_se_ratio[0] - z * spread - ratio) / spread))


def test_the_mechanism_term_controls_fall_below_the_band() -> None:
    full = point.eif_sd(point.LAW_P, point.GATED)
    dropped = point.dropped_sd(point.LAW_P, point.GATED) / full
    blip = point.blip_sd(point.LAW_P, point.GATED) / full
    assert dropped == pytest.approx(0.6136, abs=1e-4)
    assert blip == pytest.approx(0.7848, abs=1e-4)
    assert _coverage_limit(dropped) < 0.90
    assert _coverage_limit(blip) < 0.90
    for ratio in (dropped, blip):
        assert _below_band_probability(ratio, properties.CALIBRATION_REPLICATES) >= PRIMARY_PASS


def test_the_complete_case_fit_misses_its_truth() -> None:
    """Finding 2: per fit, the complete-case error exceeds 2 SE with probability >= 0.90.

    The fit is efficient on the conditioned law, so its SE at the expected count of complete
    rows is the conditioned law's bound over the root of that count.
    """
    plan = long_law.PLANS["T_a"]
    rows = properties.LONGITUDINAL_N * long_law.retention(long_law.LAW_L)
    error = long_law.complete_case_sd(long_law.LAW_L, plan) / math.sqrt(rows)
    gap = abs(properties.COMPLETE_CASE_LIMIT - properties.COMPLETE_CASE_TRUTH) / error
    probability = float(norm.cdf(gap - 2.0) + norm.cdf(-gap - 2.0))
    assert probability >= 0.90
    # The gap is a property of the dropout: the plan value on ``L0`` is the truth.
    assert abs(properties.COMPLETE_CASE_LIMIT - properties.COMPLETE_CASE_TRUTH) > 0.01


def test_the_constant_mechanism_limit_misses_the_truth() -> None:
    """No double robustness: a constant mechanism with a right regression stays biased."""
    gap = properties.CONSTANT_MECHANISM_LIMIT - properties.CONTRAST_TRUTH
    assert gap == pytest.approx(CONSTANT_GAP, abs=1e-6)
    error = point.eif_sd(point.LAW_P, point.GATED) / math.sqrt(study.REQUIREMENT_SIZES[0])
    assert abs(gap) / error > 5.0


#: The limit's offset from the truth, pinned (``constant_mechanism_limit``).
CONSTANT_GAP = -0.0647539


def test_the_constant_mechanism_limit_is_what_the_fit_reaches() -> None:
    """The population iteration against one large fit, which reads it within 4 SE."""
    frame = point.sample(point.LAW_P, 200_000, 4343)
    from sklearn.dummy import DummyClassifier

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = common.fit_point(
            frame, cross_fit=False, treatment_learner=DummyClassifier(strategy="prior")
        )
    estimate = result[point.GATED]
    assert abs(float(estimate.psi) - properties.CONSTANT_MECHANISM_LIMIT) < 4.0 * float(
        estimate.std_error
    )
    assert abs(float(estimate.psi) - properties.CONTRAST_TRUTH) > 4.0 * float(estimate.std_error)


def test_the_power_cell_is_sized_on_the_exact_law() -> None:
    power = design_power(
        properties.POWER_TRUTH, point.eif_sd(point.LAW_PW, point.GATED), properties.NULL_N
    )
    assert 0.80 <= power <= 0.90
    assert power == pytest.approx(0.8724, abs=1e-3)
    assert power_cell_pass_probability(power, properties.NULL_REPLICATES) >= PRIMARY_PASS


# ------------------------------------------------------------------------ support rows


#: The cases of the study's own two row families.
STUDY_CASES = tuple(case for case in CASES if case.row.startswith(("T35.", "G39.")))


def test_the_study_registers_one_case_for_each_of_its_six_rows() -> None:
    assert [case.row for case in STUDY_CASES] == [
        "T35.in_sample",
        "T35.cross_fitted",
        "T35.known",
        "T35.strata",
        "G39.cross_fitted",
        "G39.complete",
    ]


@pytest.mark.parametrize("case", STUDY_CASES, ids=lambda case: case.row)
def test_each_case_lands_on_its_row(case: Any) -> None:
    """The study's own fits classify onto the rows they stand for."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = case.build()
        landed = classify_key(built.key, built.learners)
    assert landed.row is not None and landed.row.id == case.row


# ---------------------------------------------------------------------- the pilot sizing

#: Family P's rung pilot, 400 draws per rung (``pilot_p.py``, ``pilot_p_summary.csv``): the
#: largest absolute standardized bias over the two primary contrasts, in sample and
#: cross-fitted.
RUNG_PILOT: dict[int, float] = {
    1_000: 0.1643,
    2_000: 0.1357,
    4_000: 0.1033,
    8_000: 0.1176,
    16_000: 0.0538,
}
#: The calibration pilot at :data:`~tests.studies.canonical_categorical_incremental.
#: POINT_CALIBRATION_N` (``pilot_cal_p.py``): per cell, the draws, the SE ratio, the relative
#: spread of the reported SE and the standardized bias.
CALIBRATION_PILOT: dict[str, tuple[int, float, float, float]] = {
    "point_contrast_t_a": (8_400, 0.9984, 0.0171, -0.0796),
    "point_contrast_t_b": (8_400, 1.0025, 0.0165, 0.0465),
    "point_known_t_a": (8_400, 1.0031, 0.0193, 0.0010),
    "point_stratified_t_a_v1": (8_400, 0.9946, 0.0317, -0.0642),
}
#: The longitudinal calibration cells at ``LONGITUDINAL_N`` (``pilot_l.py``, 1,600 draws).
LONGITUDINAL_CALIBRATION_PILOT: dict[str, tuple[int, float, float, float]] = {
    "longitudinal_dropout_contrast_t_a": (1_600, 1.0089, 0.0086, 0.0144),
    "longitudinal_contrast_t_a": (1_600, 1.0157, 0.0085, 0.0357),
}
#: The two necessity controls in the same draws: the SE ratio and the SE's relative spread.
NECESSITY_PILOT = {"mechanism_term_dropped": (0.6140, 0.0194), "binary_blip": (0.7847, 0.0222)}
#: The constant-outcome cells of the rung pilot, ``ate_ipsi[T_a vs natural course]``: the
#: draws and the standardized bias at each requirement size.
ROBUSTNESS_PILOT: dict[int, tuple[int, float]] = {8_000: (400, 0.0064), 16_000: (400, -0.0690)}
#: The null and power cells of the rung pilot at ``NULL_N``: draws and rejection rate.
NULL_PILOT = (400, 0.0475)
POWER_PILOT = (400, 0.8625)


def test_the_primary_size_is_the_smallest_unbiased_rung() -> None:
    """``X43-plan.md`` 5.3: the smallest rung with every primary contrast's bias <= 0.25."""
    unbiased = [n for n, bias in RUNG_PILOT.items() if bias <= SIZING_BIAS]
    assert min(unbiased) == study.POINT_PILOT_N
    assert study.PRIMARY_N >= study.POINT_PILOT_N


def test_the_softmax_moves_the_mechanism_out_of_fold_only() -> None:
    """Plan rev 2 finding 1, read live: the cross-fitted softmax moves the mechanism in every
    stratum; the in-sample fit, whose saturated mechanism already solves the score, cannot."""
    frame = point.sample(point.LAW_P, 4_000, 4345, strata=True)
    plain = frame.drop(columns="V")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        crossfit = common.fit_point(plain, cross_fit=True)
        in_sample = common.fit_point(plain, cross_fit=False)
        stratified = common.fit_point(frame, cross_fit=True, strata=True)
    assert common.mechanism_movement(crossfit) > 1e-4
    assert common.mechanism_movement(in_sample) < 1e-12
    targeted = np.asarray(stratified.fluctuations["ipsi"].mechanism.propensity, dtype=float)
    initial = np.asarray(stratified.nuisance.propensity.values, dtype=float)
    stratum = frame["V"].to_numpy()
    for value in (0, 1):
        rows = stratum == value
        assert np.max(np.abs(targeted[rows] - initial[rows])) > 1e-4, value


def _pass(replicates: int, ratio: float, noise: float, bias: float = 0.0) -> float:
    """The chance a cell passes the calibration rule at true ratio and standardized bias."""
    z = float(norm.ppf(0.975))
    coverage = float(norm.cdf(z * ratio - bias) - norm.cdf(-z * ratio - bias))
    return calibration_pass_probability(
        replicates, true_se_ratio=ratio, coverage=coverage, se_noise=noise
    )


def family_pass_probability(replicates: int, arms: dict[str, dict[str, Any]]) -> float:
    r"""The arm-pooled predictive pass probability of every gated calibration cell.

    The rule of ``tests/unit/test_sequential_baselines_design.py``.  Each arm's true ratio is
    :math:`N(m_a, s^2 / k_a)` with :math:`s^2` the within-arm variance pooled over the arms
    and :math:`k_a` the arm's distinct laws; each cell adds :math:`N(0, \tau^2)`, with
    :math:`\tau^2` the 80% chi-square upper bound of :math:`s^2` less the mean pilot noise.
    """
    deviations, degrees, noise_variance, means = 0.0, 0, [], {}
    for arm, cells in arms.items():
        ratios = np.array([cells[c][1] for c in cells])
        means[arm] = float(ratios.mean())
        deviations += float(((ratios - ratios.mean()) ** 2).sum())
        degrees += len(cells) - 1
        for k, ratio, noise, _ in cells.values():
            noise_variance.append(ratio**2 * (1.0 / (2.0 * (k - 1)) + noise**2 / k))
    pooled = deviations / degrees
    upper = degrees * pooled / float(chi2.ppf(1.0 - TAU_LEVEL, degrees))
    tau = math.sqrt(max(0.0, upper - float(np.mean(noise_variance))))
    nodes, weights = np.polynomial.hermite_e.hermegauss(41)
    weights = weights / weights.sum()
    total = 1.0
    for arm, cells in arms.items():
        shared = math.sqrt(pooled / ARM_LAWS[arm])
        chance = 0.0
        for node, weight in zip(nodes, weights, strict=True):
            mean = means[arm] + shared * node
            joint = 1.0
            for _, _, noise, bias in cells.values():
                joint *= (
                    sum(
                        w * _pass(replicates, mean + tau * x, noise, bias)
                        for x, w in zip(nodes, weights, strict=True)
                    )
                    if tau > 0
                    else _pass(replicates, mean, noise, bias)
                )
            chance += weight * joint
        total *= chance
    return float(total)


#: The distinct laws each arm's calibration cells draw from.
ARM_LAWS = {"point": 1, "longitudinal": 2}


def test_the_calibration_replicates_follow_the_sizing_rule() -> None:
    """``R`` is the smallest multiple of 400 with family-wise pass chance >= 0.95.

    The arms are the point cells (one law) and the longitudinal cells (laws ``L`` and
    ``L0``).
    """
    arms = {"point": CALIBRATION_PILOT, "longitudinal": LONGITUDINAL_CALIBRATION_PILOT}
    replicates = properties.CALIBRATION_REPLICATES
    assert replicates % REPLICATE_GRAIN == 0
    assert family_pass_probability(replicates, arms) >= FAMILY_FLOOR
    assert family_pass_probability(replicates - REPLICATE_GRAIN, arms) < FAMILY_FLOOR
    assert {*CALIBRATION_PILOT, *LONGITUDINAL_CALIBRATION_PILOT} == set(
        properties.CALIBRATION_LABELS
    )


def test_the_pilot_reads_the_exact_control_ratios() -> None:
    """The pilot's necessity controls sit on the exact SD ratios and fail at the declared R."""
    full = point.eif_sd(point.LAW_P, point.GATED)
    exact = {
        "mechanism_term_dropped": point.dropped_sd(point.LAW_P, point.GATED) / full,
        "binary_blip": point.blip_sd(point.LAW_P, point.GATED) / full,
    }
    for control, (ratio, noise) in NECESSITY_PILOT.items():
        assert ratio == pytest.approx(exact[control], abs=0.01)
        assert (
            _below_band_probability(ratio, properties.CALIBRATION_REPLICATES, noise) >= PRIMARY_PASS
        )


def _interval_pass_probability(rate: float, replicates: int, accept: Any) -> float:
    """The chance that the exact 99% interval of a binomial rate satisfies ``accept``."""
    level = Margins().confidence_level
    passing = [
        k
        for k in range(replicates + 1)
        if accept(clopper_pearson(k, replicates, confidence_level=level))
    ]
    return float(binom.pmf(passing, replicates, rate).sum())


def test_the_rate_cells_pass_at_their_declared_sizes() -> None:
    """Each pilot rate moves one Monte Carlo SE toward failure before it is read."""
    margins = Margins()
    k, rate = NULL_PILOT
    rejection = rate + math.sqrt(rate * (1.0 - rate) / k)
    ceiling = margins.alpha + margins.type_i_margin
    assert (
        _interval_pass_probability(
            rejection, properties.NULL_REPLICATES, lambda i: i.high <= ceiling
        )
        >= PRIMARY_PASS
    )
    k, rate = POWER_PILOT
    power = rate - math.sqrt(rate * (1.0 - rate) / k)
    assert power_cell_pass_probability(power, properties.NULL_REPLICATES) >= PRIMARY_PASS


def test_the_single_robustness_pair_passes_with_the_declared_chance() -> None:
    """Both sizes pass the bias equivalence with joint predictive chance >= 0.95.

    The cell passes when the 99% interval of the mean error, in units of the empirical SD,
    lies inside the margin.  The true bias is normal about the pilot's with the pilot's
    Monte Carlo error, and the study adds its own.
    """
    assert tuple(ROBUSTNESS_PILOT) == study.REQUIREMENT_SIZES
    replicates = properties.REQUIREMENT_REPLICATES
    margin = Margins().standardized_bias
    half = float(student.ppf(0.995, replicates - 1)) / math.sqrt(replicates)
    joint = 1.0
    for draws, bias in ROBUSTNESS_PILOT.values():
        spread = math.sqrt(1.0 / draws + 1.0 / replicates)
        joint *= float(
            norm.cdf((margin - half - bias) / spread) - norm.cdf((-margin + half - bias) / spread)
        )
    assert joint >= PRIMARY_PASS


# ------------------------------------------------------------- the longitudinal fit


def _empirical(target: Any, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The cell masses and outcome means of ``frame`` on ``target``'s axes.

    After a dropout the later axes are structural zeros, which the sample writes as missing.
    """
    names = [axis.name for axis in target.axes]
    cells = np.nan_to_num(frame[names].to_numpy(dtype=float), nan=0.0).astype(int)
    index = np.ravel_multi_index(cells.T, target.shape)
    counts = np.bincount(index, minlength=int(np.prod(target.shape))).astype(float)
    observed = ~frame["Y"].isna().to_numpy()
    totals = np.bincount(
        index[observed], weights=frame["Y"].to_numpy()[observed], minlength=counts.size
    )
    seen = np.bincount(index[observed], minlength=counts.size)
    means = np.where(seen > 0, totals / np.where(seen > 0, seen, 1), 0.0)
    return (counts / counts.sum()).reshape(target.shape), means.reshape(target.shape)


@pytest.mark.parametrize("target", [long_law.LAW_L, long_law.LAW_L0], ids=["L", "L0"])
def test_the_one_fold_fit_is_the_empirical_functional(target: Any) -> None:
    """``fit_longitudinal``'s regimens, level keys and columns reach the declared estimand.

    One fold with saturated learners makes every nuisance the empirical law's, so the
    estimate is the g-formula of the empirical law (Kennedy 2019, Algorithm 2).  A level key
    read against the wrong level, or a node tilted with the other plan's factors, moves it.
    """
    frame = long_law.sample(target, 3_000, 4343)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = common.fit_longitudinal(frame, target, n_folds=1)
    probs, means = _empirical(target, frame)
    for label, plan in long_law.PLANS.items():
        expected = float(np.real(long_law.functional(target, probs, plan, mu=means)))
        assert float(result[f"ey_regimen[{label}]"].psi) == pytest.approx(expected, abs=1e-9)
    swapped = {"T_a": long_law.PLANS["T_b"], "T_b": long_law.PLANS["T_a"]}
    expected = float(np.real(long_law.functional(target, probs, swapped["T_a"], mu=means)))
    assert abs(float(result["ey_regimen[T_a]"].psi) - expected) > 0.1


def test_the_five_fold_fit_reads_its_truth() -> None:
    """The cross-fitted fit of law ``L`` lands within 4 SE of its truth at n = 20,000."""
    frame = long_law.sample(long_law.LAW_L, 20_000, 4344)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = common.fit_longitudinal(frame, long_law.LAW_L, n_folds=common.N_FOLDS)
    estimate = result[properties.LONG_CONTRAST]
    truth = properties.TRUTH["longitudinal_dropout_contrast_t_a"]
    assert abs(float(estimate.psi) - truth) < 4.0 * float(estimate.std_error)
    assert float(estimate.std_error) * math.sqrt(20_000) == pytest.approx(
        properties.EFFICIENCY_SD["longitudinal_dropout_contrast_t_a"], rel=0.1
    )


#: The standardized bias of every primary row in every pilot: ``(n, draws, bias)``.  Family P
#: reads its 400-draw rungs (``pilot_p_summary.csv``).  Family L reads the 400-draw rungs at
#: 2k and 4k, the 2,400-draw pilot at 8k (``pilot_l_cal_summary.csv``) and the 1,600-draw
#: pilot at 16k (``pilot_l_16k_summary.csv``).
POINT_BIAS: dict[tuple[str, str], tuple[tuple[int, int, float], ...]] = {
    ("crossfit", "ate_ipsi[T_a vs natural course]"): (
        (1000, 400, -0.1643),
        (2000, 400, -0.0560),
        (4000, 400, -0.1033),
        (8000, 400, 0.0260),
        (16000, 400, -0.0538),
    ),
    ("crossfit", "ate_ipsi[T_b vs natural course]"): (
        (1000, 400, 0.0881),
        (2000, 400, 0.0511),
        (4000, 400, 0.0538),
        (8000, 400, -0.0277),
        (16000, 400, 0.0368),
    ),
    ("crossfit", "ey_ipsi[T_a]"): (
        (1000, 400, -0.0122),
        (2000, 400, -0.0047),
        (4000, 400, -0.0084),
        (8000, 400, 0.0168),
        (16000, 400, -0.0208),
    ),
    ("crossfit", "ey_ipsi[T_b]"): (
        (1000, 400, 0.0978),
        (2000, 400, 0.0405),
        (4000, 400, 0.0612),
        (8000, 400, -0.0055),
        (16000, 400, 0.0197),
    ),
    ("crossfit", "ey_ipsi[natural course]"): (
        (1000, 400, 0.0530),
        (2000, 400, 0.0173),
        (4000, 400, 0.0352),
        (8000, 400, 0.0062),
        (16000, 400, 0.0012),
    ),
    ("in_sample", "ate_ipsi[T_a vs natural course]"): (
        (1000, 400, 0.0839),
        (2000, 400, 0.1357),
        (4000, 400, 0.0289),
        (8000, 400, 0.1176),
        (16000, 400, 0.0127),
    ),
    ("in_sample", "ate_ipsi[T_b vs natural course]"): (
        (1000, 400, -0.0699),
        (2000, 400, -0.0793),
        (4000, 400, -0.0380),
        (8000, 400, -0.0887),
        (16000, 400, -0.0068),
    ),
    ("in_sample", "ey_ipsi[T_a]"): (
        (1000, 400, 0.0845),
        (2000, 400, 0.0706),
        (4000, 400, 0.0475),
        (8000, 400, 0.0532),
        (16000, 400, 0.0063),
    ),
    ("in_sample", "ey_ipsi[T_b]"): (
        (1000, 400, 0.0236),
        (2000, 400, -0.0158),
        (4000, 400, 0.0189),
        (8000, 400, -0.0316),
        (16000, 400, -0.0021),
    ),
    ("in_sample", "ey_ipsi[natural course]"): (
        (1000, 400, 0.0530),
        (2000, 400, 0.0173),
        (4000, 400, 0.0352),
        (8000, 400, 0.0062),
        (16000, 400, 0.0012),
    ),
}
LONGITUDINAL_BIAS: dict[tuple[str, str], tuple[tuple[int, int, float], ...]] = {
    ("crossfit_L", "ate_regimen[T_a vs natural course]"): (
        (2000, 400, 0.1353),
        (4000, 400, 0.0441),
        (8000, 2400, 0.0768),
        (16000, 1600, 0.0144),
    ),
    ("crossfit_L", "ate_regimen[T_b vs natural course]"): (
        (2000, 400, -0.0752),
        (4000, 400, -0.0493),
        (8000, 2400, -0.0671),
        (16000, 1600, -0.0312),
    ),
    ("crossfit_L", "ey_regimen[T_a]"): (
        (2000, 400, 0.0501),
        (4000, 400, 0.0445),
        (8000, 2400, 0.0286),
        (16000, 1600, 0.0237),
    ),
    ("crossfit_L", "ey_regimen[T_b]"): (
        (2000, 400, -0.0609),
        (4000, 400, -0.0073),
        (8000, 2400, -0.0530),
        (16000, 1600, -0.0016),
    ),
    ("crossfit_L", "ey_regimen[natural course]"): (
        (2000, 400, -0.0177),
        (4000, 400, 0.0248),
        (8000, 2400, -0.0099),
        (16000, 1600, 0.0180),
    ),
    ("crossfit_L0", "ate_regimen[T_a vs natural course]"): (
        (2000, 400, 0.1292),
        (4000, 400, 0.1105),
        (8000, 2400, 0.0738),
        (16000, 1600, 0.0357),
    ),
    ("crossfit_L0", "ate_regimen[T_b vs natural course]"): (
        (2000, 400, -0.1538),
        (4000, 400, -0.1211),
        (8000, 2400, -0.0550),
        (16000, 1600, -0.0512),
    ),
    ("crossfit_L0", "ey_regimen[T_a]"): (
        (2000, 400, 0.0833),
        (4000, 400, -0.0115),
        (8000, 2400, 0.0169),
        (16000, 1600, 0.0597),
    ),
    ("crossfit_L0", "ey_regimen[T_b]"): (
        (2000, 400, -0.0786),
        (4000, 400, -0.1464),
        (8000, 2400, -0.0598),
        (16000, 1600, 0.0168),
    ),
    ("crossfit_L0", "ey_regimen[natural course]"): (
        (2000, 400, 0.0214),
        (4000, 400, -0.0686),
        (8000, 2400, -0.0221),
        (16000, 1600, 0.0471),
    ),
}


def _primary_pass(
    table: dict[tuple[str, str], tuple[tuple[int, int, float], ...]], n: int
) -> float:
    r"""The joint predictive chance that every primary row passes the bias gate at ``n``.

    Each row's standardized bias decays as :math:`c/\sqrt{n}` (a second-order remainder),
    with :math:`c` the least-squares fit over the pilots, each pilot bias weighted by its draws.
    A row passes when the 99% t interval of its mean error over ``PRIMARY_REPLICATES`` lies
    inside the margin.
    """
    replicates = study.PRIMARY_REPLICATES
    margin = Margins().standardized_bias
    half = float(student.ppf(0.995, replicates - 1)) / math.sqrt(replicates)
    joint = 1.0
    for pilots in table.values():
        x = np.array([1.0 / math.sqrt(size) for size, _, _ in pilots])
        k = np.array([float(draws) for _, draws, _ in pilots])
        y = np.array([bias for _, _, bias in pilots])
        c = float((k * x * y).sum() / (k * x * x).sum())
        se = 1.0 / math.sqrt(float((k * x * x).sum()))
        f = 1.0 / math.sqrt(n)
        spread = math.sqrt((se * f) ** 2 + 1.0 / replicates)
        joint *= float(
            norm.cdf((margin - half - c * f) / spread) - norm.cdf((-margin + half - c * f) / spread)
        )
    return joint


def test_the_primary_size_passes_the_bias_gate() -> None:
    """``PRIMARY_N`` is the smallest rung with joint primary pass chance >= 0.95.

    The plan's rung rule (bias <= 0.25 in a pilot) passes at every rung, 1,000 for P and 2,000
    for L, but there the gate's joint chance is far below 0.95.
    """
    for table, pilot in (
        (POINT_BIAS, study.POINT_PILOT_N),
        (LONGITUDINAL_BIAS, study.LONGITUDINAL_PILOT_N),
    ):
        assert len(table) == 10
        rungs = sorted({size for pilots in table.values() for size, _, _ in pilots})
        worst = {
            size: max(abs(b) for pilots in table.values() for s, _, b in pilots if s == size)
            for size in rungs
        }
        assert pilot == min(size for size in rungs if worst[size] <= SIZING_BIAS)
        assert _primary_pass(table, pilot) < PRIMARY_PASS

    def joint(n: int) -> float:
        return _primary_pass(POINT_BIAS, n) * _primary_pass(LONGITUDINAL_BIAS, n)

    assert joint(study.PRIMARY_N) >= PRIMARY_PASS
    rungs = sorted({size for pilots in LONGITUDINAL_BIAS.values() for size, _, _ in pilots})
    assert all(joint(n) < PRIMARY_PASS for n in rungs if n < study.PRIMARY_N)


# --------------------------------------------------------- law L by a second route


def _longhand_joint(stay: bool) -> dict[tuple[int, int, int, int], float]:
    """``P(W, A1, L2, A2)`` of law ``L`` from its conditionals, in loops.

    ``stay=False`` prevents dropout (the target law).  ``stay=True`` keeps only the rows that
    never drop out and renormalises: the law a complete-case fit sees.
    """
    out: dict[tuple[int, int, int, int], float] = {}
    for w in (0, 1):
        for a1 in range(3):
            for l2 in (0, 1):
                for a2 in range(3):
                    v = {"W": w, "A1": a1, "C1": 1, "L2": l2, "A2": a2, "C2": 1}
                    mass = 0.5
                    mass *= long_law._conditional(long_law.LAW_L.axes[1], v)[a1]
                    mass *= long_law._conditional(long_law.LAW_L.axes[3], v)[l2]
                    mass *= long_law._conditional(long_law.LAW_L.axes[4], v)[a2]
                    if stay:
                        mass *= long_law._conditional(long_law.LAW_L.axes[2], v)[1]
                        mass *= long_law._conditional(long_law.LAW_L.axes[5], v)[1]
                    out[(w, a1, l2, a2)] = mass
    total = sum(out.values())
    return {key: value / total for key, value in out.items()}


def _longhand_gformula(joint: dict[tuple[int, int, int, int], float], plan: Any) -> float:
    """The tilted g-formula of ``plan`` read off ``joint`` by sums over its cells."""
    f1, f2 = plan[0][1], plan[1][1]

    def mass(**fixed: int) -> float:
        names = ("w", "a1", "l2", "a2")
        return sum(
            p
            for key, p in joint.items()
            if all(key[names.index(name)] == value for name, value in fixed.items())
        )

    def tilt(g: list[float], factors: tuple[float, ...]) -> list[float]:
        total = sum(f * p for f, p in zip(factors, g, strict=True))
        return [f * p / total for f, p in zip(factors, g, strict=True)]

    value = 0.0
    for w in (0, 1):
        pw = mass(w=w)
        q1 = tilt([mass(w=w, a1=a) / pw for a in range(3)], f1)
        for a1 in range(3):
            pa = mass(w=w, a1=a1)
            for l2 in (0, 1):
                pl = mass(w=w, a1=a1, l2=l2)
                q2 = tilt([mass(w=w, a1=a1, l2=l2, a2=a) / pl for a in range(3)], f2)
                for a2 in range(3):
                    v = {"W": w, "A1": a1, "L2": l2, "A2": a2}
                    value += pw * q1[a1] * (pl / pa) * q2[a2] * long_law._mean(v)
    return value


def test_the_longitudinal_truths_by_a_second_route() -> None:
    """Finding 4: the plan values and the complete-case limit, longhand from the law's
    conditionals, against :func:`functional` on both laws."""
    target = _longhand_joint(stay=False)
    for plan in long_law.PLANS.values():
        expected = _longhand_gformula(target, plan)
        for law_ in (long_law.LAW_L, long_law.LAW_L0):
            assert long_law.truth(law_, plan) == pytest.approx(expected, abs=1e-12)
    complete = _longhand_gformula(_longhand_joint(stay=True), long_law.PLANS["T_a"])
    assert pytest.approx(complete, abs=1e-12) == properties.COMPLETE_CASE_LIMIT
    assert properties.TRUTH["longitudinal_contrast_t_a"] == pytest.approx(
        _longhand_gformula(target, long_law.PLANS["T_a"])
        - _longhand_gformula(target, long_law.PLANS[long_law.REFERENCE]),
        abs=1e-12,
    )


@pytest.mark.parametrize("target", [long_law.LAW_L, long_law.LAW_L0], ids=["L", "L0"])
def test_the_longitudinal_mechanism_term_carries_its_share(target: Any) -> None:
    """Finding 5: the curve without its mechanism terms (the treatment conditionals frozen at
    the truth) falls below the calibration band, so the L calibration cells see an omission."""
    plan, other = long_law.PLANS["T_a"], long_law.PLANS[long_law.REFERENCE]
    full = long_law.eif_sd(target, plan, other=other)
    frozen = long_law.eif_sd(target, plan, other=other, frozen_treatment=True)
    assert frozen / full < Margins().calibration_se_ratio[0] - 0.1
    assert 1.0 - (frozen / full) ** 2 >= 0.30


def test_each_longitudinal_scenario_publishes_its_own_bound() -> None:
    """Finding 7: laws L and L0 share their truths and not their bounds."""
    for scenario, (family, target, _) in study.SCENARIOS.items():
        if family != "L":
            continue
        for name in study.LONGITUDINAL_ESTIMANDS:
            assert study.EFFICIENCY_SD[f"{scenario}/{name}"] == pytest.approx(
                study.longitudinal_eif_sd(target, name), abs=1e-12
            )
    name = properties.LONG_CONTRAST
    assert study.EFFICIENCY_SD[f"longitudinal_folds_5/{name}"] == pytest.approx(
        properties.EFFICIENCY_SD["longitudinal_contrast_t_a"], abs=1e-12
    )


# ------------------------------------------------------------ mechanism targeting

#: The ``mechanism_targeting`` pilot at n = 4,000 (``pilot_mt.py``, 4,000 draws): per positive
#: cell the draws, the SE ratio, the SE's relative spread and the standardized bias; for the
#: control the draws and the standardized bias.
TARGETING_PILOT: dict[str, tuple[int, float, float, float]] = {
    "point_contrast_t_a__mechanism_targeted": (4_000, 1.0020, 0.0170, -0.0524),
    "point_stratified_t_a_v1__mechanism_targeted": (4_000, 0.9947, 0.0309, -0.0191),
}
CONTROL_PILOT = (4_000, 1.2694)


def _exact_frame(*, strata: bool = False) -> pd.DataFrame:
    """Law ``P`` itself as weighted rows: each cell twice, at ``mu - 1`` and ``mu + 1``."""
    rows = []
    for w in range(len(point.P_W)):
        for a, label in enumerate(point.LABELS):
            for sign in (-1.0, 1.0):
                rows.append(
                    {
                        "W": float(w),
                        "A": label,
                        "Y": point.MU[w, a] + sign,
                        "weight": point.P_W[w] * point.G[w, a] / 2.0,
                        "V": float(point.STRATA_OF_W[w]),
                    }
                )
    frame = pd.DataFrame(rows)
    return frame if strata else frame.drop(columns="V")


def _exact_fit(*, strata: bool = False) -> Any:
    """The in-sample TMLE on the exact law, with the tilted mechanism of the family."""
    from cleverly.estimators import TMLE

    estimator = TMLE(
        incremental=common.incrementals(),
        outcome_learner=common.CellMeans(),
        treatment_learner=common.TiltedFrequencies(properties.TARGETING_LAMBDA),
        cross_fit=False,
        q_bounds=common.Q_BOUNDS,
        simultaneous=False,
        max_iter=common.MAX_ITER,
        tol=common.TOL,
        random_state=0,
    )
    roles: dict[str, Any] = {"covariates": ["W", "V"] if strata else ["W"], "weights": "weight"}
    if strata:
        roles["strata"] = ["V"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator.fit(
            _exact_frame(strata=strata), outcome="Y", treatment="A", **roles
        ).single()


def _scale() -> float:
    return point.eif_sd(point.LAW_P, point.GATED) / math.sqrt(study.POINT_CALIBRATION_N)


def test_the_tilted_mechanism_is_a_first_order_error() -> None:
    """The control's exact limit is 1.18 SD at n = 4,000 and linear in the tilt."""
    truth = properties.CONTRAST_TRUTH
    assert (properties.OUTCOME_ONLY_LIMIT - truth) / _scale() == pytest.approx(1.18045, abs=1e-4)
    half = point.truth(
        point.PointLaw(
            "half", point.P_W, common.tilted_mechanism(properties.TARGETING_LAMBDA / 2), point.MU
        ),
        point.GATED,
    )
    assert (half - truth) / (properties.OUTCOME_ONLY_LIMIT - truth) == pytest.approx(0.5, abs=0.01)
    assert pytest.approx(2.0 / math.sqrt(4_000), abs=1e-15) == properties.TARGETING_LAMBDA


def test_only_the_mechanism_fluctuation_removes_the_error_on_the_exact_law() -> None:
    """On the exact law the TMLE reaches the truth, and the outcome-only plug-in stays at the
    control's exact limit.  The stratified softmax reaches the stratum truth."""
    result = _exact_fit()
    assert abs(float(result[point.GATED].psi) - properties.CONTRAST_TRUTH) / _scale() < 1e-3
    outcome_only = incremental_estimates(result, targeted_outcome=True)[point.GATED]
    assert outcome_only == pytest.approx(properties.OUTCOME_ONLY_LIMIT, abs=1e-9)
    stratified = _exact_fit(strata=True)
    assert float(stratified[properties.STRATIFIED].psi) == pytest.approx(
        properties.TRUTH["point_stratified_t_a_v1"], abs=1e-6
    )


def test_an_uncentred_mechanism_covariate_fails_the_positive_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M6 on the exact law moves the TMLE past the bias margin of the positive cell.

    M5 (one-versus-rest update) leaves the estimate within 2e-4 SD here; it is read by the
    simplex witness of ``tests/unit/test_categorical_incremental_mutations.py``.
    """
    mutate(
        monkeypatch,
        mechanism_module,
        mechanism_module,
        "_covariate_general",
        [("predictions[:, :, None] - mixture[:, None, :]", "predictions[:, :, None]")],
    )
    result = _exact_fit()
    bias = (float(result[point.GATED].psi) - properties.CONTRAST_TRUTH) / _scale()
    assert abs(bias) > Margins().standardized_bias + 0.1


def _targeting_pass(replicates: int) -> float:
    """Each positive cell passes bias equivalence and the efficiency band, and the control's
    bias interval lies beyond the margin: the joint predictive chance at ``replicates``."""
    margin = Margins().standardized_bias
    half = float(student.ppf(0.995, replicates - 1)) / math.sqrt(replicates)
    z = float(norm.ppf(0.995))
    low, high = properties.EFFICIENCY_RATIO_BAND
    joint = 1.0
    for draws, ratio, noise, bias in TARGETING_PILOT.values():
        spread = math.sqrt(1.0 / draws + 1.0 / replicates)
        unbiased = norm.cdf((margin - half - bias) / spread) - norm.cdf(
            (-margin + half - bias) / spread
        )
        width = ratio * math.sqrt(1.0 / (2.0 * (replicates - 1)) + noise**2 / replicates)
        known = ratio * math.sqrt(1.0 / (2.0 * (draws - 1)) + noise**2 / draws)
        total = math.hypot(width, known)
        banded = norm.cdf((high - z * width - ratio) / total) - norm.cdf(
            (low + z * width - ratio) / total
        )
        joint *= float(unbiased * banded)
    draws, bias = CONTROL_PILOT
    spread = math.sqrt(1.0 / draws + 1.0 / replicates)
    joint *= float(norm.cdf((bias - margin - half) / spread))
    return joint


def test_the_mechanism_targeting_replicates_follow_the_sizing_rule() -> None:
    """``R`` is the smallest multiple of 400 with family pass chance >= 0.95, and at least 2,000."""
    replicates = properties.TARGETING_REPLICATES
    assert replicates % REPLICATE_GRAIN == 0
    assert _targeting_pass(replicates) >= PRIMARY_PASS
    assert replicates == 2_000 or _targeting_pass(replicates - REPLICATE_GRAIN) < PRIMARY_PASS


@pytest.mark.parametrize("case", STUDY_CASES, ids=lambda case: case.row)
def test_each_case_cites_the_line_of_its_fit(case: Any) -> None:
    """Finding 9: a case's ``file:line`` is the line that fits, not merely a line in range."""
    path, _, line = case.source.rpartition(":")
    text = (study.ROOT / path).read_text().splitlines()[int(line) - 1]
    assert "common.fit_point(" in text or "common.fit_longitudinal(" in text, case.source
