"""Focused contracts for the stacked arm-indexed MAR CV-TMLE comparator study.

The registered verdicts are recomputed by ``tests/unit/test_method_evidence.py``.  These
checks cover what that module cannot see: the laws' exact truths and efficient curves, the
oracle learners, the misspecified-nuisance limits the robustness cells are designed around,
the reference payload, the R adapter's pinned settings, and the joint-coverage rule.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, stats

from tests import discrete_law_mar as mar
from tests.studies import canonical_mar_arm_indexed_cvtmle as study
from tests.studies import mar_arm_indexed_cvtmle_properties as properties
from tests.studies import mar_arm_indexed_laws as laws
from tests.studies.evidence.registry import ROOT
from tests.studies.missing_outcome_study_helpers import efficiency_sd, probabilities

LAWS = tuple(laws.LAWS.values())
IDS = tuple(laws.LAWS)


def test_the_two_arm_curves_match_the_observed_data_gateaux_derivative() -> None:
    """The closed-form covariance against :mod:`tests.discrete_law_mar`'s complex step."""
    law = laws.LAWS["l1"]
    cells = probabilities(mar.Q, g=mar.G, pi=mar.PI, p_w=mar.P_W)
    for name in laws.ESTIMANDS["l1"]:
        assert laws.efficiency_sd(law, name) == pytest.approx(efficiency_sd(cells, name), rel=1e-12)


def _three_arm_cells(law: laws.Law) -> np.ndarray:
    """``P(W = w, A = a, K = k)`` with ``k`` observed zero, observed one, or missing."""
    out = np.empty((3, 3, 3))
    for w, a in itertools.product(range(3), range(3)):
        base = law.p_w[w] * law.g[w, a]
        out[w, a] = base * np.array(
            [law.pi[w, a] * (1 - law.mu[w, a]), law.pi[w, a] * law.mu[w, a], 1 - law.pi[w, a]]
        )
    return out


def _three_arm_functional(cells: np.ndarray, law: laws.Law, name: str) -> complex:
    p_w = cells.sum(axis=(1, 2))
    q = cells[:, :, 1] / (cells[:, :, 0] + cells[:, :, 1])
    psi = [(p_w * q[:, a]).sum() for a in range(3)]
    arm, reference = laws._arm_of(law, name)
    head = laws.stem(name)
    if reference is None:
        return psi[arm]
    if head == "ate":
        return psi[arm] - psi[reference]
    if head == "rr":
        return np.log(psi[arm]) - np.log(psi[reference])
    odds = [np.log(value / (1 - value)) for value in psi]
    return odds[arm] - odds[reference]


def test_the_three_arm_curves_match_a_complex_step_gateaux_derivative() -> None:
    """The per-arm residual terms share no row, so the arms' curves covary only through W."""
    law = laws.LAWS["l3"]
    cells = _three_arm_cells(law)
    points = list(itertools.product(range(3), range(3), range(3)))
    step = 1e-30
    for name in laws.ESTIMANDS["l3"]:
        curve = []
        for point in points:
            mass = np.zeros_like(cells, dtype=complex)
            mass[point] = 1.0
            perturbed = (1.0 - 1j * step) * cells.astype(complex) + 1j * step * mass
            curve.append(np.imag(_three_arm_functional(perturbed, law, name)) / step)
        deviation = math.sqrt(
            sum(cells[point] * value**2 for point, value in zip(points, curve, strict=True))
        )
        assert laws.efficiency_sd(law, name) == pytest.approx(deviation, rel=1e-10)


def test_the_continuous_outcome_variance_is_the_scaled_beta_variance() -> None:
    law = laws.LAWS["l2"]
    expected = (laws.UPPER - laws.LOWER) ** 2 * stats.beta.var(
        laws.PHI * law.mu, laws.PHI * (1 - law.mu)
    )
    np.testing.assert_allclose(law.variance(), expected, rtol=1e-12)


@pytest.mark.parametrize("law", LAWS, ids=IDS)
def test_the_sampler_realises_the_declared_tables(law: laws.Law) -> None:
    frame = laws.sample(law, 400_000, 7)
    w = frame["W"].to_numpy(dtype=int)
    np.testing.assert_allclose(np.bincount(w) / len(frame), law.p_w, atol=0.005)
    for level, column in itertools.product(range(3), range(law.arms)):
        rows = (w == level) & (frame["A"] == law.labels[column]).to_numpy()
        assert rows.mean() / (w == level).mean() == pytest.approx(law.g[level, column], abs=0.01)
        assert frame.loc[rows, "Delta"].mean() == pytest.approx(law.pi[level, column], abs=0.02)
        observed = frame.loc[rows & (frame["Delta"] == 1).to_numpy(), "Y"]
        assert observed.notna().all()
        expected = law.offset + law.span * law.mu[level, column]
        assert observed.mean() == pytest.approx(expected, abs=0.03 * law.span)
    assert frame.loc[frame["Delta"] == 0, "Y"].isna().all()
    if law.continuous:
        assert frame["Y"].dropna().between(laws.LOWER, laws.UPPER).all()


@pytest.mark.parametrize("law", LAWS, ids=IDS)
def test_the_oracle_learners_reach_the_fit_as_the_law_s_own_tables(law: laws.Law) -> None:
    frame, _ = study.draw_scenario(law.scenario, 400, 0)
    result = study.fit_cleverly(
        frame,
        law,
        outcome_learner=laws.LawOutcome(law),
        treatment_learner=laws.LawTreatment(law),
        missingness_learner=laws.LawResponse(law),
    )
    w = frame["W"].to_numpy(dtype=int)
    nuisance = result.nuisance
    for code, arm in enumerate(result.data.arm_codes):
        column = law.column(code)
        assert result.data.arm_label(arm) == law.codes[code]
        np.testing.assert_array_equal(nuisance.propensity.arm(arm), law.g[w, column])
        np.testing.assert_array_equal(nuisance.missingness[:, code], law.pi[w, column])
        np.testing.assert_allclose(nuisance.outcome.arms[arm], law.mu[w, column], atol=1e-15)


@pytest.mark.parametrize("law", LAWS, ids=IDS)
def test_one_wrong_nuisance_leaves_the_limit_at_the_truth(law: laws.Law) -> None:
    truth = laws.truths(law)
    wrong = laws.wrong_tables(law)
    for kwargs in ({"mu": wrong["mu"]}, {"g": wrong["g"]}, {"pi": wrong["pi"]}):
        limit = laws.targeted_limit(law, **kwargs)
        for name in truth:
            assert limit[name] == pytest.approx(truth[name], abs=1e-12)


@pytest.mark.parametrize("law", LAWS, ids=IDS)
def test_the_bias_control_limit_moves_every_contrast(law: laws.Law) -> None:
    """The control's design: a wrong ``Q`` beside a wrong ``pi`` moves each ATE materially.

    Every contrast moves by more than a tenth of the outcome's range, which is at least
    four times its efficient standard error at the study's size.
    """
    truth = laws.truths(law)
    wrong = laws.wrong_tables(law)
    limit = laws.targeted_limit(law, mu=wrong["mu"], pi=wrong["pi"])
    for name in laws.ESTIMANDS[law.key]:
        if laws.stem(name) != "ate":
            continue
        moved = abs(limit[name] - truth[name])
        assert moved > 0.1 * law.span
        assert moved > 4 * laws.efficiency_sd(law, name) / math.sqrt(study.PRIMARY_N)


@pytest.mark.parametrize("law", LAWS, ids=IDS)
def test_pointwise_intervals_read_jointly_undercover_in_the_limit(law: laws.Law) -> None:
    """The pointwise joint control is designed to under-cover: its limit is below 0.90.

    It is a negative control for the coverage cell, not a check on the band. A wrong band
    that is still wide enough, such as a Sidak band, passes the coverage cell beside it.
    The critical-value check below is the one that fails such a band.
    """
    assert laws.pointwise_joint_coverage(law, draws=200_000) < 0.90


def test_the_reference_payload_serializes_the_fitted_predictions_and_folds() -> None:
    law = laws.LAWS["l3"]
    frame, _ = study.draw_scenario(law.scenario, 300, 0)
    result = study.fit_cleverly(frame, law)
    sample = study.reference_sample(frame, result, scenario=law.scenario, replicate=0)
    nuisance = result.nuisance
    for code, arm in enumerate(result.data.arm_codes):
        expected = nuisance.scaler.unscale_levels(nuisance.outcome.arms[arm])
        np.testing.assert_array_equal(sample[f"q{code}"], expected)
        np.testing.assert_array_equal(sample[f"g{code}"], nuisance.propensity.arm(arm))
        np.testing.assert_array_equal(sample[f"pi{code}"], nuisance.missingness[:, code])
    np.testing.assert_array_equal(sample["fold"], nuisance.folds.assignment)
    np.testing.assert_array_equal(sample["A"], np.asarray(result.data.treatment, dtype=int))
    assert set(sample["fold"]) == set(range(study.N_FOLDS))


def test_a_two_arm_payload_leaves_the_third_arm_empty() -> None:
    law = laws.LAWS["l2"]
    frame, _ = study.draw_scenario(law.scenario, 300, 0)
    result = study.fit_cleverly(frame, law)
    sample = study.reference_sample(frame, result, scenario=law.scenario, replicate=0)
    assert sample[["q2", "g2", "pi2"]].isna().all().all()
    np.testing.assert_allclose(sample["g0"] + sample["g1"], 1.0, atol=1e-12)


def test_the_r_adapter_pins_the_audited_settings() -> None:
    source = (study.STUDY.artifacts / "run_study.R").read_text(encoding="utf-8")
    for pinned in (
        "q_bounds <- c(-2, 8)",
        "gbound <- 0.001",
        "alpha <- 0.9995",
        'three_arm_labels <- c("high", "low", "mid")',
        'family = if (continuous) "gaussian" else "binomial"',
        'fluctuation = "logistic"',
        "Qbounds = if (continuous) q_bounds else c(0, 1)",
        "cvQinit = FALSE",
        "prescreenW.g = FALSE",
        "target.gwt = FALSE",
        "B = 1",
        "evalATT = FALSE",
        "a <- rep(1, n)",
        "delta <- as.numeric(frame$A == code & frame$Delta == 1)",
        "y <- ifelse(delta == 1, frame$Y, NA_real_)",
        "y <- plant(y, delta == 0)",
        "fit$estimates$IC$IC.EY1",
        "covariance <- stats::cov(influence) / n",
        "tmle:::.initStage1(",
    ):
        assert pinned in source, pinned
    assert study.R_GBOUND == 0.001
    assert (study.STUDY.artifacts / "probe_scale_workaround.R").exists()
    assert not list(study.STUDY.artifacts.glob("run_*probe*.R"))


def test_the_scale_probe_failure_refuses_publication() -> None:
    passing = pd.DataFrame(
        {
            "scenario": [law.scenario for law in LAWS if law.continuous],
            "passed": [True, True],
        }
    )
    assert study.scientific_failures({"scale-probe.csv": passing})["scale-workaround probe"].empty
    failing = passing.assign(passed=[True, False])
    assert len(study.scientific_failures({"scale-probe.csv": failing})["scale-workaround probe"])
    partial = passing.iloc[:1]
    assert "scale-workaround probe coverage" in study.scientific_failures(
        {"scale-probe.csv": partial}
    )


def test_ratio_property_rows_are_on_the_log_scale() -> None:
    law = laws.LAWS["l1"]
    frame, truth = study.draw_scenario(law.scenario, 400, 1)
    result = study.fit_cleverly(frame, law)
    row = properties._estimand_row(
        "interval_calibration",
        "l1_rr__learned_nuisances",
        "positive",
        0,
        400,
        1,
        "rr",
        truth["rr"],
        result["rr"],
    )
    assert row["truth"] == pytest.approx(math.log(truth["rr"]))
    assert row["estimate"] == pytest.approx(float(result["rr"].log_psi))
    assert row["std_error"] == pytest.approx(float(result["rr"].std_error))


def test_the_joint_rows_read_the_package_band() -> None:
    """The band row covers exactly when the max-t statistic is inside its critical value."""
    law = laws.LAWS["l3"]
    for replicate in range(4):
        frame, truth = study.draw_scenario(law.scenario, 400, replicate)
        result = study.fit_cleverly(frame, law, simultaneous=True)
        band, pointwise = properties._joint_rows(
            law, result, truth, replicate=replicate, n=400, requested=4
        )
        assert band["std_error"] == result.simultaneous.critical_value
        assert band["std_error"] > pointwise["std_error"]
        assert band["covered"] == int(band["estimate"] <= band["std_error"])
        assert pointwise["covered"] == int(pointwise["estimate"] <= pointwise["std_error"])


def _committed_property_rows() -> pd.DataFrame:
    path = study.STUDY.artifact("property-replicates.csv.gz")
    if not path.exists():  # pragma: no cover - only before the first registration
        pytest.skip("the study has no committed property rows")
    return pd.read_csv(path)


def test_a_band_at_the_pointwise_critical_value_fails_its_cell() -> None:
    """A deliberate mutation: the band's rows replaced by the pointwise joint rows.

    The mutation is what a band that reused the pointwise critical value would publish.  Its
    joint coverage falls out of the calibration band, and no other verdict moves.
    """
    rows = _committed_property_rows()
    published = properties.summarize_properties(rows).set_index(["property", "cell"])
    mutated = rows.copy()
    for key in laws.LAWS:
        band = (mutated["property"] == "simultaneous_coverage") & (
            mutated["cell"] == f"{key}__simultaneous_band"
        )
        control = (mutated["property"] == "simultaneous_coverage") & (
            mutated["cell"] == f"{key}__pointwise_joint_control"
        )
        mutated.loc[band, "covered"] = mutated.loc[control, "covered"].to_numpy()
    summary = properties.summarize_properties(mutated).set_index(["property", "cell"])
    changed = [("simultaneous_coverage", f"{key}__simultaneous_band") for key in laws.LAWS]
    assert not summary.loc[changed, "passed"].any()
    untouched = summary.index.drop(changed)
    assert summary.loc[untouched, "passed"].equals(published.loc[untouched, "passed"])


#: The multiplier draws and the level behind each committed band critical value.
BAND_DRAWS = 1000
BAND_LEVEL = 0.95

#: How far the mean committed critical value of a law may sit from its limiting target.
#: Each critical value is ``np.quantile`` of ``BAND_DRAWS`` multiplier max-t draws, so its
#: Monte Carlo standard deviation is ``sqrt(0.95 * 0.05 / 1000) / f`` for the max-t density
#: ``f`` at the quantile.  Here ``f`` is 0.125 to 0.131, which gives 0.053 to 0.055, and the
#: standard error of a mean over 2,000 replications is 0.0012 to 0.0013.  The tolerance is
#: four of those.  A band that ignores the correlation (Sidak or Bonferroni) sits at least
#: 0.08 above the target, sixteen tolerances away.
CRITICAL_VALUE_TOLERANCE = 0.005


def _sphere(rank: int, points: int) -> np.ndarray:
    """A deterministic near-uniform grid on the unit circle or the unit sphere."""
    index = np.arange(points) + 0.5
    if rank == 2:
        angle = 2.0 * np.pi * index / points
        return np.column_stack([np.cos(angle), np.sin(angle)])
    height = 1.0 - 2.0 * index / points
    angle = np.pi * (1.0 + math.sqrt(5.0)) * index
    radius = np.sqrt(1.0 - height**2)
    return np.column_stack([radius * np.cos(angle), radius * np.sin(angle), height])


def _max_t_cdf(law: laws.Law, points: int = 100_000) -> Any:
    """``P(max_j |Z_j| <= c)`` for ``Z`` with the law's limiting influence-curve correlation.

    The correlation has rank two (two arms) or three (three arms), because every estimand
    is a function of the arm means.  Write ``Z = A U`` with ``U`` standard normal in that
    rank.  Along a direction ``u`` on the unit sphere, ``max_j |Z_j| <= c`` holds out to
    radius ``c / max_j |a_j . u|``, so the probability is the chi distribution function at
    that radius, averaged over the sphere.  The grid is converged to 1e-5 in the quantile.
    """
    covariance = laws.influence_covariance(law)
    scale = np.sqrt(np.diag(covariance))
    values_, vectors = np.linalg.eigh(covariance / np.outer(scale, scale))
    keep = values_ > 1e-10 * values_.max()
    factor = vectors[:, keep] * np.sqrt(values_[keep])
    rank = factor.shape[1]
    assert rank == law.arms
    reach = np.abs(_sphere(rank, points) @ factor.T).max(axis=1)
    return lambda c: float(np.mean(stats.chi.cdf(c / reach, rank)))


def _quantile(cdf: Any, level: float) -> float:
    return float(optimize.brentq(lambda c: cdf(c) - level, 1.0, 5.0, xtol=1e-10))


def _band_target(law: laws.Law) -> float:
    """Where a 1,000-draw ``np.quantile`` of the limiting max-t statistic lands on average.

    ``np.quantile`` interpolates between order statistics at position ``(B - 1) p``, and
    the ``k``-th of ``B`` uniform order statistics has mean ``k / (B + 1)``.  So the
    expected probability level is ``((B - 1) p + 1) / (B + 1) = 0.94910``, not 0.95, and
    the correct band's mean critical value sits 0.007 below the limiting quantile.
    """
    cdf = _max_t_cdf(law)
    level = ((BAND_DRAWS - 1) * BAND_LEVEL + 1) / (BAND_DRAWS + 1)
    return _quantile(cdf, level)


def _band_critical_values(rows: pd.DataFrame, key: str) -> pd.Series:
    band = (rows["property"] == "simultaneous_coverage") & (
        rows["cell"] == f"{key}__simultaneous_band"
    )
    return rows.loc[band, "std_error"]


def _band_misses(rows: pd.DataFrame) -> list[str]:
    """The laws whose mean critical value is outside the tolerance of its target."""
    misses = []
    for key, law in laws.LAWS.items():
        target = _band_target(law)
        if abs(_band_critical_values(rows, key).mean() - target) > CRITICAL_VALUE_TOLERANCE:
            misses.append(key)
    return misses


def test_the_study_bands_use_the_draws_the_target_assumes() -> None:
    law = laws.LAWS["l1"]
    configured = study.method(
        law,
        outcome_learner=None,
        treatment_learner=None,
        missingness_learner=None,
        simultaneous=True,
    ).estimator_kwargs()
    assert configured["n_multiplier"] == BAND_DRAWS
    assert configured["multiplier_kind"] == "rademacher"


@pytest.mark.parametrize("law", LAWS, ids=IDS)
def test_the_tolerance_is_four_monte_carlo_standard_errors(law: laws.Law) -> None:
    """The tolerance's justification, recomputed from each law's max-t density."""
    cdf = _max_t_cdf(law)
    quantile = _quantile(cdf, BAND_LEVEL)
    density = (cdf(quantile + 1e-4) - cdf(quantile - 1e-4)) / 2e-4
    replicates = len(_band_critical_values(_committed_property_rows(), law.key))
    spread = math.sqrt(BAND_LEVEL * (1 - BAND_LEVEL) / BAND_DRAWS) / density
    standard_errors = CRITICAL_VALUE_TOLERANCE / (spread / math.sqrt(replicates))
    assert 3.5 < standard_errors < 4.5


def test_the_band_critical_values_match_the_limiting_max_t_quantile() -> None:
    """The coverage cell cannot see a band that ignores the correlation; this check can.

    A Sidak or Bonferroni band covers at least the nominal rate, so it passes three of the
    four joint-coverage cells.  Its critical value is what gives it away.
    """
    assert _band_misses(_committed_property_rows()) == []


@pytest.mark.parametrize("rule", ["sidak", "bonferroni"])
def test_a_diagonal_covariance_band_fails_the_critical_value_check(rule: str) -> None:
    """A deliberate mutation: each band critical value replaced by a diagonal-covariance one."""
    rows = _committed_property_rows().copy()
    for key in laws.LAWS:
        count = len(laws.ESTIMANDS[key])
        tail = 1 - BAND_LEVEL ** (1 / count) if rule == "sidak" else (1 - BAND_LEVEL) / count
        band = (rows["property"] == "simultaneous_coverage") & (
            rows["cell"] == f"{key}__simultaneous_band"
        )
        rows.loc[band, "std_error"] = float(stats.norm.ppf(1 - tail / 2))
    assert _band_misses(rows) == list(laws.LAWS)
    for key, law in laws.LAWS.items():
        assert _band_critical_values(rows, key).mean() - _band_target(law) > 0.08


#: The largest ``|cleverly - R| / |targeting move|`` the parity check admits in any
#: replication and estimand.  Both implementations target the same stitched predictions, so
#: their initial estimates agree to rounding and their targeted estimates differ only through
#: where each fluctuation solver stops.  cleverly iterates to a relative score of ``1e-10``.
#: R ``tmle`` fits the fluctuation with ``glm`` at its default ``epsilon = 1e-8``, a relative
#: change in deviance.  The deviance is quadratic in the coefficient near its minimum, so that
#: stop resolves the coefficient, and with it the move, to about ``sqrt(1e-8) = 1e-4`` of its
#: scale.  The bound allows a hundredfold over that for a move whose coefficient is close to
#: zero.  A missing targeting step has ratio 1 (the initial estimates agree), and a step wrong
#: by a fraction ``f`` has a ratio near ``f``, so the bound still sits two orders below both.
TARGETING_RATIO_BOUND = 1e-2


def _paired_primary_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The committed primary rows of each implementation, aligned by replication and estimand."""
    rows = pd.read_csv(study.STUDY.artifact("replicates.csv.gz"))
    key = ["scenario", "replicate", "estimand"]
    subject = rows[rows["implementation"] == study.STUDY.implementation].set_index(key)
    reference = rows[rows["implementation"] == study.STUDY.reference].set_index(key)
    assert len(subject) == len(reference) == len(rows) // 2
    return subject.sort_index(), reference.loc[subject.sort_index().index]


def _targeting_ratio(
    estimate: pd.Series, subject: pd.DataFrame, reference: pd.DataFrame
) -> pd.Series:
    move = (subject["estimate"] - subject["initial_estimate"]).abs()
    assert (move > 0).all(), "a replication has no targeting move"
    return (estimate - reference["estimate"]).abs() / move


def test_the_r_parity_reproduces_every_targeting_move() -> None:
    """The paired tests cannot see a missing targeting step; this ratio can.

    The saturated primary trees leave the initial plug-in almost unbiased, so an estimate
    that skipped targeting still passes every paired equivalence test and every truth test.
    Each replication's difference from R is instead compared with the size of its own
    targeting move.
    """
    subject, reference = _paired_primary_rows()
    shared = (subject["initial_estimate"] - reference["initial_estimate"]).abs()
    assert shared.max() < 1e-12
    ratio = _targeting_ratio(subject["estimate"], subject, reference)
    assert ratio.max() < TARGETING_RATIO_BOUND, ratio.idxmax()


def test_a_missing_targeting_step_fails_the_parity_ratio() -> None:
    """A deliberate mutation: cleverly's reported estimate replaced by its initial plug-in."""
    subject, reference = _paired_primary_rows()
    ratio = _targeting_ratio(subject["initial_estimate"], subject, reference)
    assert (ratio >= TARGETING_RATIO_BOUND).all()
    assert ratio.min() > 0.9


def test_the_study_directory_is_the_registered_one() -> None:
    assert study.STUDY.artifacts == ROOT / "tests" / "canonical" / "tmle_mar_arm_indexed_cvtmle"
    assert study.STUDY.artifacts.joinpath("regenerate.py").exists()
