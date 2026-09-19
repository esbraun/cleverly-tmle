"""Focused contracts for the stacked arm-indexed MAR CV-TMLE comparator study.

The registered verdicts are recomputed by ``tests/unit/test_method_evidence.py``.  These
checks cover what that module cannot see: the laws' exact truths and efficient curves, the
oracle learners, the misspecified-nuisance limits the robustness cells are designed around,
the reference payload, the R adapter's pinned settings, and the joint-coverage rule.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

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
    """The band's control is designed to fail: its limiting joint coverage is below 0.90."""
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


def test_the_study_directory_is_the_registered_one() -> None:
    assert study.STUDY.artifacts == ROOT / "tests" / "canonical" / "tmle_mar_arm_indexed_cvtmle"
    assert study.STUDY.artifacts.joinpath("regenerate.py").exists()
