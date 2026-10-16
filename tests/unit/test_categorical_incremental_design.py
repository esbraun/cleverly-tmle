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

from cleverly._support import classify_key
from cleverly._support_rows import PENDING_ROWS
from tests import discrete_law_categorical_ipsi_longitudinal as long_law
from tests import discrete_law_categorical_ipsi_point as point
from tests.studies import canonical_categorical_incremental as study
from tests.studies import categorical_incremental_common as common
from tests.studies import categorical_incremental_properties as properties
from tests.studies.canonical_full_refit_bootstrap import calibration_pass_probability
from tests.studies.evidence.inference import clopper_pearson
from tests.studies.evidence.property_verdicts import design_power, power_cell_pass_probability
from tests.studies.evidence.registry import Margins, registered
from tests.studies.support_cases import PENDING_CASES

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


@pytest.mark.parametrize(
    "case",
    PENDING_CASES,
    ids=lambda case: case.row,
)
def test_each_case_lands_on_its_pending_row_only(case: Any) -> None:
    """The study's own fits classify onto the pending rows they stand for, and onto no
    registered row: until registration such a fit is labelled experimental."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = case.build()
        landed = classify_key(built.key, built.learners, rows=PENDING_ROWS)
        registered_row = classify_key(built.key, built.learners).row
    assert landed.row is not None and landed.row.id == case.row
    assert registered_row is None


# ---------------------------------------------------------------------- the pilot sizing

#: Family P's rung pilot, 400 draws per rung (``pilot_p.py``, ``pilot_p_summary.csv``): the
#: largest absolute standardized bias over the two primary contrasts, in sample and
#: cross-fitted, and the median mechanism movement of the cross-fitted and in-sample fits.
RUNG_PILOT: dict[int, tuple[float, float, float]] = {
    1_000: (0.1643, 0.0074, 0.0),
    2_000: (0.1357, 0.0037, 0.0),
    4_000: (0.1033, 0.0019, 0.0),
    8_000: (0.1176, 0.0009, 0.0),
    16_000: (0.0538, 0.0004, 0.0),
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
    unbiased = [n for n, (bias, _, _) in RUNG_PILOT.items() if bias <= SIZING_BIAS]
    assert min(unbiased) == study.POINT_PILOT_N
    assert study.PRIMARY_N >= study.POINT_PILOT_N


def test_the_softmax_witness_moves_out_of_fold_only() -> None:
    """Finding 1: the cross-fitted fits move the mechanism; the in-sample fits cannot."""
    for crossfit, in_sample in ((c, i) for _, c, i in RUNG_PILOT.values()):
        assert crossfit > 0.0 and in_sample == 0.0


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

    Provisional: only the point arm is piloted.  The longitudinal arm joins the rule with
    its pilot (stage 2), and the declared ``R`` can only rise.
    """
    arms = {"point": CALIBRATION_PILOT}
    replicates = properties.CALIBRATION_REPLICATES
    assert replicates % REPLICATE_GRAIN == 0
    assert family_pass_probability(replicates, arms) >= FAMILY_FLOOR
    assert family_pass_probability(replicates - REPLICATE_GRAIN, arms) < FAMILY_FLOOR
    point_labels = {label for label in properties.CALIBRATION_LABELS if label.startswith("point")}
    assert set(CALIBRATION_PILOT) == point_labels


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


#: The rung pilots' standardized bias of every primary row (``pilot_p_summary.csv``, 400 draws at
#: 1k, 2k, 4k, 8k and 16k; ``pilot_l_summary.csv``, 400 draws at 2k, 4k, 8k and 16k).
POINT_BIAS: dict[tuple[str, str], tuple[float, ...]] = {
    ("crossfit", "ate_ipsi[T_a vs natural course]"): (-0.1643, -0.0560, -0.1033, 0.0260, -0.0538),
    ("crossfit", "ate_ipsi[T_b vs natural course]"): (0.0881, 0.0511, 0.0538, -0.0277, 0.0368),
    ("crossfit", "ey_ipsi[T_a]"): (-0.0122, -0.0047, -0.0084, 0.0168, -0.0208),
    ("crossfit", "ey_ipsi[T_b]"): (0.0978, 0.0405, 0.0612, -0.0055, 0.0197),
    ("crossfit", "ey_ipsi[natural course]"): (0.0530, 0.0173, 0.0352, 0.0062, 0.0012),
    ("in_sample", "ate_ipsi[T_a vs natural course]"): (0.0839, 0.1357, 0.0289, 0.1176, 0.0127),
    ("in_sample", "ate_ipsi[T_b vs natural course]"): (-0.0699, -0.0793, -0.0380, -0.0887, -0.0068),
    ("in_sample", "ey_ipsi[T_a]"): (0.0845, 0.0706, 0.0475, 0.0532, 0.0063),
    ("in_sample", "ey_ipsi[T_b]"): (0.0236, -0.0158, 0.0189, -0.0316, -0.0021),
    ("in_sample", "ey_ipsi[natural course]"): (0.0530, 0.0173, 0.0352, 0.0062, 0.0012),
}
LONGITUDINAL_BIAS: dict[tuple[str, str], tuple[float, ...]] = {
    ("crossfit_L", "ate_regimen[T_a vs natural course]"): (0.1353, 0.0441, 0.0729, 0.0462),
    ("crossfit_L", "ate_regimen[T_b vs natural course]"): (-0.0752, -0.0493, -0.0698, -0.1297),
    ("crossfit_L", "ey_regimen[T_a]"): (0.0501, 0.0445, 0.0502, 0.0741),
    ("crossfit_L", "ey_regimen[T_b]"): (-0.0609, -0.0073, -0.0278, -0.0231),
    ("crossfit_L", "ey_regimen[natural course]"): (-0.0177, 0.0248, 0.0165, 0.0582),
    ("crossfit_L0", "ate_regimen[T_a vs natural course]"): (0.1292, 0.1105, 0.0095, 0.0499),
    ("crossfit_L0", "ate_regimen[T_b vs natural course]"): (-0.1538, -0.1211, 0.0050, -0.0610),
    ("crossfit_L0", "ey_regimen[T_a]"): (0.0833, -0.0115, -0.1017, 0.0418),
    ("crossfit_L0", "ey_regimen[T_b]"): (-0.0786, -0.1464, -0.1238, -0.0192),
    ("crossfit_L0", "ey_regimen[natural course]"): (0.0214, -0.0686, -0.1173, 0.0199),
}
POINT_RUNGS = (1_000, 2_000, 4_000, 8_000, 16_000)
LONGITUDINAL_RUNGS = (2_000, 4_000, 8_000, 16_000)


def _primary_pass(
    table: dict[tuple[str, str], tuple[float, ...]], rungs: tuple[int, ...], n: int
) -> float:
    """The joint predictive chance that every primary row passes the bias gate at ``n``.

    Each row's standardized bias decays as :math:`c\\sqrt{n_0/n}` (a second-order remainder),
    with :math:`c` the least-squares fit over the rungs, each pilot bias of variance 1/400.
    A row passes when the 99% t interval of its mean error over ``PRIMARY_REPLICATES`` lies
    inside the margin.
    """
    replicates = study.PRIMARY_REPLICATES
    margin = Margins().standardized_bias
    half = float(student.ppf(0.995, replicates - 1)) / math.sqrt(replicates)
    x = np.sqrt(rungs[0] / np.asarray(rungs, dtype=float))
    joint = 1.0
    for biases in table.values():
        c = float(x @ np.asarray(biases)) / float(x @ x)
        se = 1.0 / math.sqrt(400.0 * float(x @ x))
        f = math.sqrt(rungs[0] / n)
        spread = math.sqrt((se * f) ** 2 + 1.0 / replicates)
        joint *= float(
            norm.cdf((margin - half - c * f) / spread) - norm.cdf((-margin + half - c * f) / spread)
        )
    return joint


def test_the_primary_size_passes_the_bias_gate() -> None:
    """``PRIMARY_N`` is the smallest rung with joint primary pass chance >= 0.95.

    The plan's rung rule (bias <= 0.25 at 400 draws) passes at every rung, 1,000 for P and
    2,000 for L, but there the gate's joint chance is far below 0.95.
    """
    for table, rungs, pilot in (
        (POINT_BIAS, POINT_RUNGS, study.POINT_PILOT_N),
        (LONGITUDINAL_BIAS, LONGITUDINAL_RUNGS, study.LONGITUDINAL_PILOT_N),
    ):
        assert len(table) == 10
        assert pilot == min(
            n for i, n in enumerate(rungs) if max(abs(b[i]) for b in table.values()) <= SIZING_BIAS
        )
        assert _primary_pass(table, rungs, pilot) < PRIMARY_PASS

    def joint(n: int) -> float:
        return _primary_pass(POINT_BIAS, POINT_RUNGS, n) * _primary_pass(
            LONGITUDINAL_BIAS, LONGITUDINAL_RUNGS, n
        )

    assert joint(study.PRIMARY_N) >= PRIMARY_PASS
    smaller = [n for n in LONGITUDINAL_RUNGS if n < study.PRIMARY_N]
    assert all(joint(n) < PRIMARY_PASS for n in smaller)
