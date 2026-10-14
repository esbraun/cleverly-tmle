"""The declared design of the DR-TMLE one-inconsistent-nuisance witness study (RM40b).

Every number here was fixed before the registered run: the budgets, the seeds, the two laws
and their truths, the population limits of the wrong regressions, the predicted sign of each
first-order control, and the pass probability of each gated rule.  The predicted signs and
sizes come from the population conditioning term, by quadrature, and not from a fit.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from scipy.stats import norm

from tests.studies import drtmle_one_wrong as study
from tests.studies import drtmle_one_wrong_properties as properties
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.registry import ROOT, Margins, registered

pytestmark = pytest.mark.xdist_group("drtmle_one_wrong_design")

W1, W2, WEIGHTS = study._grid()
LADDER_SIZES = (1_000, 4_000, 16_000)


def test_the_declared_numbers() -> None:
    record = study.STUDY
    assert record.publication_policy == "reporting"
    assert record.margins == Margins()
    assert record.reference is None
    assert (study.PRIMARY_REPLICATES, study.PRIMARY_N) == (4_000, 4_000)
    assert (study.SEED, study.RESAMPLING_SEED) == (20264001, 2026400101)
    assert study.G_BOUNDS == (0.01, 0.99)
    assert (properties.PAIRED_N, properties.PAIRED_REPLICATES) == (4_000, 4_000)
    assert (properties.CROSS_FIT_N, properties.CROSS_FIT_REPLICATES) == (4_000, 3_000)
    assert (properties.LADDER, properties.LADDER_REPLICATES) == (LADDER_SIZES, 1_000)
    assert (properties.LAW_M_N, properties.LAW_M_REPLICATES) == (4_000, 1_000)
    assert (study.SLOW_G_OFFSET, study.SLOW_Q_OFFSET, study.SLOW_PI_OFFSET) == (1.0, 3.0, 2.0)
    assert properties.RED_CELL_OWNER == "RM40-conservative-resolution"


def test_the_study_is_registered_and_draws_from_its_own_seeds() -> None:
    records = {record.slug: record for record in registered()}
    assert study.STUDY.slug in records
    others = {
        seed
        for record in registered()
        if record.slug != study.STUDY.slug
        for seed in (record.seed, record.resampling_seed)
    }
    assert not {study.SEED, study.RESAMPLING_SEED} & others


def test_no_cell_is_a_power_cell_and_the_runners_exist() -> None:
    assert not {"power", "type_i_error"} & set(study.STUDY.property_cells)
    here = ROOT / "tests" / "canonical" / "drtmle_one_wrong"
    assert (here / "regenerate.py").exists()


def test_the_truths_match_quadrature_and_an_independent_monte_carlo() -> None:
    truth = study.truth()
    assert truth["ey1"] == pytest.approx(0.5422909862, abs=1e-10)
    assert truth["ey0"] == pytest.approx(0.4717756171, abs=1e-10)
    assert truth["ate"] == pytest.approx(0.0705153691, abs=1e-10)
    rng = np.random.default_rng(11)
    size = 2_000_000
    w1, w2 = rng.standard_normal((2, size))
    for arm, name in ((1.0, "ey1"), (0.0, "ey0")):
        values = study.qbar0(arm, w1, w2)
        error = float(values.std() / math.sqrt(size))
        assert abs(float(values.mean()) - truth[name]) < 5 * error


def test_the_population_limits_are_the_plans() -> None:
    """Plan 5.2 and 5.2a: the wrong outcome GLM's limits on both laws.

    The plan's values are Monte Carlo fits (2e6 draws), so they agree with the quadrature to
    their own error.  The quadrature values are pinned exactly; a 4e6-draw fit to ``Qbar0``
    reproduces them to 2e-3.
    """
    np.testing.assert_allclose(study.outcome_limit_l(), (-0.3075, 0.6732, -0.6733), atol=6e-3)
    np.testing.assert_allclose(study.outcome_limit_m(), (0.0267, 0.2936, -0.6687), atol=6e-3)
    np.testing.assert_allclose(
        study.outcome_limit_l(), (-0.30559267, 0.67308131, -0.66857797), atol=1e-7
    )
    np.testing.assert_allclose(
        study.outcome_limit_m(), (0.02489607, 0.29388157, -0.66450486), atol=1e-7
    )


def test_the_bounds_never_bind() -> None:
    """Plan 5.2: the index SD is 0.67, so ``g0`` stays inside ``(0.01, 0.99)``."""
    rng = np.random.default_rng(3)
    w1, w2 = rng.standard_normal((2, 4_000_000))
    g = study.g0(w1, w2)
    share = float(np.mean((g < study.G_BOUNDS[0]) | (g > study.G_BOUNDS[1])))
    assert share < 1e-5


def test_the_declared_cells_are_the_runners() -> None:
    declared = {(family, cell) for family, cell, _, _ in properties.declared_truths()}
    growth = {
        (property_verdicts.ONE_WRONG_FAMILY, cell)
        for cell in study.PROPERTY_CELLS[property_verdicts.ONE_WRONG_FAMILY]
        if property_verdicts.one_wrong_kind(cell) == "growth"
    }
    published = {(family, cell) for family, cells in study.PROPERTY_CELLS.items() for cell in cells}
    # A growth row is built from its ladder's rungs and has no replicate rows of its own.
    assert declared | growth == published
    assert not declared & growth


def test_every_rule_kind_is_known_and_every_control_has_a_sign() -> None:
    for cell in study.PROPERTY_CELLS[property_verdicts.ONE_WRONG_FAMILY]:
        kind = property_verdicts.one_wrong_kind(cell)
        assert kind in {"conservative", "valid", "first_order", "growth"}
        if kind in {"first_order", "growth"}:
            assert property_verdicts.one_wrong_stem(cell) in properties.PREDICTED_SIGNS
    for setting in properties.LADDERS:
        assert f"{setting}__ate__growth" in study.PROPERTY_CELLS[property_verdicts.ONE_WRONG_FAMILY]


def test_the_paired_arms_share_their_draws() -> None:
    glm = properties._sample("paired", 0, 500, "L")
    again = properties._sample("paired", 0, 500, "L")
    pd.testing.assert_frame_equal(glm, again)
    other = properties._sample(properties.CROSS_FIT, 0, 500, "L")
    assert not np.allclose(glm["W1"].to_numpy(), other["W1"].to_numpy())


# --------------------------------------------------------------- the predicted conditioning terms
def _slow_g_term(n: int) -> float:
    """``sqrt(n) R_3`` of the slow-mechanism ladder at the outcome GLM's limit, ATE."""
    beta = study.outcome_limit_l()
    offset = study.SLOW_G_OFFSET * study.slow(n)
    arms = {}
    for arm in (1.0, 0.0):
        g = study.g0(W1, W2)
        fitted = expit(0.6 * W1 + 0.3 * W2 + offset * W2)
        if arm == 0.0:
            g, fitted = 1.0 - g, 1.0 - fitted
        residual = (
            study.qbar0(arm, W1, W2)
            - expit(beta[0] + beta[1] * arm + beta[2] * W1)
            - study.reduced_outcome_limit(arm, beta, W1, W2)
        )
        arms[arm] = float((WEIGHTS * residual * (g - fitted) / g).sum())
    return math.sqrt(n) * (arms[1.0] - arms[0.0])


def _mirror_term(n: int) -> float:
    """``sqrt(n) R~_5`` of the mirror ladder at the wrong mechanism GLM's limit, ATE."""
    gbar = study.mechanism_limit_mirror()
    offset = study.SLOW_Q_OFFSET * study.slow(n)
    b0, ba, b1, b2 = study.OUTCOME_COEFFICIENTS
    direction = study.PERPENDICULAR[0] * W1 + study.PERPENDICULAR[1] * W2
    arms = {}
    for arm in (1.0, 0.0):
        weight = study.mirror_weight(arm, gbar, W1, W2)
        fitted = expit(b0 + ba * arm + b1 * W1 + b2 * W2 + offset * direction)
        arms[arm] = float((WEIGHTS * weight * (study.qbar0(arm, W1, W2) - fitted)).sum())
    return math.sqrt(n) * (arms[1.0] - arms[0.0])


def _response_terms(n: int) -> dict[str, float]:
    """``sqrt(n)`` times eq. (21) of Díaz and van der Laan (2017) per arm on law M."""
    beta = study.outcome_limit_m()
    offset = study.SLOW_PI_OFFSET * study.slow(n)
    out = {}
    for arm, name in ((1.0, "ey1"), (0.0, "ey0")):
        truth = study.pi0(arm, W1, W2)
        fitted = expit(0.4 + 0.2 * arm + 0.6 * W1 + 0.3 * W2 + offset * W2)
        residual = (
            study.qbar0(arm, W1, W2)
            - expit(beta[0] + beta[1] * arm + beta[2] * W1)
            - study.reduced_outcome_limit(arm, beta, W1, W2)
        )
        out[name] = math.sqrt(n) * float((WEIGHTS * residual * (truth - fitted) / fitted).sum())
    return out


def _law_m_sd(arm: float, name: str) -> float:
    """``sqrt(n)`` SD of the arm's curve at the outcome limit and the true response mechanism."""
    beta = study.outcome_limit_m()
    q0 = study.qbar0(arm, W1, W2)
    fitted = expit(beta[0] + beta[1] * arm + beta[2] * W1)
    centred = fitted - study.truth()[name]
    probability = study.LAW_M_TREATMENT * study.pi0(arm, W1, W2)
    second = (
        centred**2
        + (q0 * (1.0 - q0) + (q0 - fitted) ** 2) / probability
        + 2.0 * centred * (q0 - fitted)
    )
    return math.sqrt(float((WEIGHTS * second).sum()))


def test_the_predicted_signs_are_the_population_terms() -> None:
    """Every first-order control's declared sign is its population term's, at every rung."""
    for n in LADDER_SIZES:
        assert np.sign(_slow_g_term(n)) == properties.PREDICTED_SIGNS["q_wrong_g_slow__ate"]
        assert np.sign(_mirror_term(n)) == properties.PREDICTED_SIGNS["q_slow_g_wrong__ate"]
    for name, value in _response_terms(properties.LAW_M_N).items():
        assert np.sign(value) == properties.PREDICTED_SIGNS[f"mar_q_wrong_pi_slow__{name}"]


def test_the_ladder_terms_grow_on_the_root_n_scale() -> None:
    """Rule G's premise: ``sqrt(n)`` times an ``n^(-1/3)`` term grows by ``16^(1/6)``."""
    for term in (_slow_g_term, _mirror_term):
        values = [abs(term(n)) for n in LADDER_SIZES]
        assert values[0] < values[1] < values[2]
        assert values[2] / values[0] == pytest.approx(16 ** (1 / 6), rel=0.05)


def test_the_law_m_term_is_at_least_two_margins() -> None:
    """Plan 5.2a: ``|term| / SD >= 2 x 0.25`` at each arm at ``n = 4000``."""
    terms = _response_terms(properties.LAW_M_N)
    for arm, name in ((1.0, "ey1"), (0.0, "ey0")):
        assert abs(terms[name]) / _law_m_sd(arm, name) >= 2 * Margins().standardized_bias


def test_the_construction_offset_reaches_the_fit() -> None:
    """The learners read the design ``[A, W1, W2]`` in that order, and add their offset."""
    rng = np.random.default_rng(5)
    frame = study.draw_law_l(200_000, 17)
    design = frame[["A", "W1", "W2"]].to_numpy()
    n = 4_000
    learner = study.outcome_learner(study.SETTINGS["q_slow_g_wrong"], n).fit(design, frame["Y"])
    np.testing.assert_allclose(learner.beta_, study.OUTCOME_COEFFICIENTS, atol=0.05)
    probe = rng.standard_normal((5, 3))
    delta = study.SLOW_Q_OFFSET * study.slow(n)
    plain = learner.beta_[0] + probe @ learner.beta_[1:]
    shifted = np.log(learner.predict_proba(probe)[:, 1] / learner.predict_proba(probe)[:, 0])
    expected = delta * (study.PERPENDICULAR[0] * probe[:, 1] + study.PERPENDICULAR[1] * probe[:, 2])
    np.testing.assert_allclose(shifted - plain, expected, atol=1e-10)
    treatment = study.mechanism_learner(study.SETTINGS["q_wrong_g_slow"], n)
    treatment.fit(design[:, 1:], frame["A"])
    np.testing.assert_allclose(treatment.beta_, (0.0, *study.MECHANISM_INDEX), atol=0.03)
    wrong = study.mechanism_learner(study.SETTINGS["q_slow_g_wrong"], n)
    assert wrong.columns == (1,)


# -------------------------------------------------------------------- the pass probabilities
def _conservative_pass(ratio: float, replicates: int, inflation: float) -> float:
    """Plan 5.6: ``P(rho_hat - 2.576 sd > 1.07)``, ``sd = inflation * rho / sqrt(2R)``."""
    sd = inflation * ratio / math.sqrt(2 * replicates)
    threshold = Margins().calibration_se_ratio[1] + 2.576 * sd
    return float(norm.cdf((ratio - threshold) / sd))


def test_the_conservative_cells_are_sized() -> None:
    in_sample = properties.PLANNING_RATIO["q_wrong_g_glm"]
    assert in_sample == pytest.approx(1.1441 - 1.2816 * 0.0148, abs=1e-12)
    assert _conservative_pass(in_sample, properties.PAIRED_REPLICATES, 1.07) >= 0.92
    assert _conservative_pass(in_sample, properties.PAIRED_REPLICATES, 1.08) >= 0.91
    cross = properties.PLANNING_RATIO["q_wrong_g_glm_cf"]
    assert _conservative_pass(cross, properties.CROSS_FIT_REPLICATES, 0.96) >= 0.98
    assert _conservative_pass(cross, properties.CROSS_FIT_REPLICATES, 1.08) >= 0.95
    for setting, ratio in properties.PLANNING_RATIO.items():
        coverage = 2 * norm.cdf(1.96 * ratio) - 1
        assert coverage == pytest.approx(properties.PLANNING_COVERAGE[setting], abs=1e-3)


def test_the_controls_are_sized() -> None:
    """Plan 5.6: the least negative measured bias of each control against its rule."""
    margin = Margins().standardized_bias
    for bias, replicates in (
        (-0.66, properties.LADDER_REPLICATES),
        (-0.46, properties.LAW_M_REPLICATES),
    ):
        sd = 1.0 / math.sqrt(replicates)
        assert norm.cdf((abs(bias) - margin - 2.576 * sd) / sd) > 0.999
    growth_sd = math.sqrt(2.0 / properties.LADDER_REPLICATES) * 1.35
    assert norm.cdf((0.46 - 2.576 * growth_sd) / growth_sd) > 0.999


def test_the_declared_cells_are_the_cells_a_run_publishes() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = properties.generate_property_rows(n_jobs=1, budget=3)
    declared = {(family, cell): truth for family, cell, _, truth in properties.declared_truths()}
    assert {tuple(key) for key in rows.groupby(["property", "cell"]).groups} == set(declared)
    for (family, cell), truth in declared.items():
        selected = rows.loc[(rows["property"] == family) & (rows["cell"] == cell)]
        np.testing.assert_allclose(selected["truth"], truth, rtol=1e-12, atol=1e-12)
        assert int(selected["failed_replicates"].max()) == 0
    assert set(properties.DIAGNOSTIC_COLUMNS) <= set(rows.columns)
