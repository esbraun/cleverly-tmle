"""The declared laws of the registered study ``ctmle-derivative``, checked before any run.

These checks read the laws alone.  The penalty pair, the seeds, the cells and the budget join
them when the study modules exist.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from tests.studies import ctmle_derivative_laws as laws
from tests.studies.canonical_full_refit_bootstrap import calibration_pass_probability
from tests.studies.evidence.property_verdicts import (
    MINIMUM_POWER,
    design_power,
    power_cell_pass_probability,
)
from tests.studies.evidence.registry import Margins

pytestmark = pytest.mark.xdist_group("ctmle_derivative_design")


@pytest.mark.parametrize("law", list(laws.LAWS.values()), ids=lambda law: law.name)
def test_each_truth_matches_a_monte_carlo(law: laws.DerivativeLaw) -> None:
    """The quadrature arm means against ``200,000`` draws, to four Monte Carlo errors."""
    w = np.random.default_rng(20263001).standard_normal((200_000, 4))
    truth = law.arm_means()
    for a in range(law.k):
        values = law.outcome_probability(a, w)
        error = values.std() / np.sqrt(values.size)
        assert abs(values.mean() - truth[a]) < 4.0 * error


def test_the_sharp_null_has_a_zero_contrast_and_an_outcome_that_reads_w() -> None:
    assert laws.SHARP_NULL.truth()["ate"] == pytest.approx(0.0, abs=1e-12)
    assert laws.SHARP_NULL.slope == laws.PRIMARY.slope
    assert laws.SHARP_NULL.scores == laws.PRIMARY.scores


def test_the_power_cell_is_sized_from_the_exact_law() -> None:
    contrast = laws.ALTERNATIVE.truth()["ate"]
    power = design_power(contrast, laws.ALTERNATIVE_EIF_SD, laws.POWER_N)
    assert power >= 0.90
    assert power >= MINIMUM_POWER
    assert power_cell_pass_probability(power, laws.POWER_REPLICATES) >= 0.95


def test_the_declared_efficient_sd_reproduces() -> None:
    """``10^5`` draws reproduce the ``10^6``-draw constant to four of their own errors."""
    sd, error = laws.ALTERNATIVE.eif_sd(100_000, 2)
    assert abs(sd - laws.ALTERNATIVE_EIF_SD) < 4.0 * error


def test_each_calibration_cell_is_sized_for_its_pass_probability() -> None:
    margins = Margins()
    probability = calibration_pass_probability(
        laws.CALIBRATION_REPLICATES,
        true_se_ratio=1.0,
        confidence_level=margins.confidence_level,
        se_band=margins.calibration_se_ratio,
        coverage_band=margins.calibration_coverage,
    )
    assert probability >= 0.90


def _projection_kl(law: laws.DerivativeLaw) -> float:
    rng = np.random.default_rng(20263003)
    w = rng.standard_normal((100_000, 4))
    g = law.mechanism(w)[:, 1]
    a = (rng.uniform(size=g.size) < g).astype(float)
    p = LogisticRegression(C=1e6, max_iter=2000, tol=1e-10).fit(w, a).predict_proba(w)[:, 1]
    return float(np.mean(g * np.log(g / p) + (1 - g) * np.log((1 - g) / (1 - p))))


def test_the_main_terms_lasso_target_is_right_on_the_primary_law_and_wrong_off_it() -> None:
    assert _projection_kl(laws.PRIMARY) < laws.PRIMARY_KL_CEILING
    assert _projection_kl(laws.OUTCOME_CORRECT) > laws.OUTCOME_CORRECT_KL_FLOOR


@pytest.mark.parametrize("law", [laws.PRIMARY, laws.THREE_ARM], ids=lambda law: law.name)
def test_the_mechanism_stays_inside_the_bounds(law: laws.DerivativeLaw) -> None:
    g = law.mechanism(np.random.default_rng(20263004).standard_normal((200_000, 4)))
    share = float(np.mean((g < laws.G_BOUNDS[0]) | (g > laws.G_BOUNDS[1])))
    assert share < laws.TRUNCATION_CEILING


def test_the_clustered_law_keeps_the_primary_marginals() -> None:
    """At ``10^5`` rows both logistic fits recover the declared coefficients to 0.05."""
    frame = laws.CLUSTERED.sample(100_000, 20263005)
    w = frame[list(laws.COVARIATES)].to_numpy()
    outcome = LogisticRegression(C=1e6, max_iter=2000).fit(
        np.column_stack([frame["A"], w]), frame["Y"]
    )
    c0, c1 = laws.PRIMARY.intercepts
    expected = [c1 - c0, 0.0, *laws.PRIMARY.slope]
    assert outcome.intercept_[0] == pytest.approx(c0, abs=0.05)
    np.testing.assert_allclose(outcome.coef_[0], expected, atol=0.05)
    treatment = LogisticRegression(C=1e6, max_iter=2000).fit(w, frame["A"])
    assert treatment.intercept_[0] == pytest.approx(0.0, abs=0.05)
    np.testing.assert_allclose(treatment.coef_[0], laws.PRIMARY.scores[0], atol=0.05)
    # The copula makes the outcomes of one cluster dependent.
    residual = frame["Y"] - outcome.predict_proba(np.column_stack([frame["A"], w]))[:, 1]
    by_cluster = residual.groupby(frame["cluster"]).sum()
    assert by_cluster.var() > 3.0 * residual.var() * laws.CLUSTERED.clusters
