"""The declared laws of the registered study ``ctmle-derivative``, checked before any run.

The laws are checked alone.  The declared penalty pair, the cells, the budgets and the
targeting-necessity threshold are checked against the study modules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from tests.studies import canonical_ctmle_derivative as study
from tests.studies import ctmle_derivative_dr_pilot as pilot
from tests.studies import ctmle_derivative_laws as laws
from tests.studies import ctmle_derivative_properties as properties
from tests.studies.canonical_full_refit_bootstrap import calibration_pass_probability
from tests.studies.evidence.property_verdicts import (
    CLUSTER_ROBUST_CONTROL_SE_CEILING,
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
    assert _projection_kl(laws.OUTCOME_CORRECT) >= pilot.KL_FLOOR


def test_the_outcome_correct_law_is_the_one_the_committed_pilot_chose() -> None:
    """The pilot assessed the whole grid, and the law's coefficient is its smallest pass."""
    frame = pd.read_csv(pilot.PILOT, float_precision="round_trip")
    assert list(frame["curvature"]) == pytest.approx(list(pilot.CURVATURE_GRID))
    assert pilot.chosen_curvature() == laws.OUTCOME_CURVATURE
    assert pilot.law_for(laws.OUTCOME_CURVATURE) == laws.OUTCOME_CORRECT
    raw = frame.drop(
        columns=[c for c in frame.columns if c.startswith("trend_")]
        + ["check_4", "check_5", "admissible"]
    )
    reselected = pilot.select(raw)
    for column in ("check_4", "check_5", "admissible"):
        assert reselected[column].tolist() == frame[column].astype(bool).tolist()
    chosen = frame.loc[np.isclose(frame["curvature"], laws.OUTCOME_CURVATURE)].iloc[0]
    assert chosen["truncation"] <= pilot.DR_TRUNCATION_CEILING
    assert abs(chosen["tmle_limit_bias"]) >= pilot.TMLE_LIMIT_FLOOR * chosen["sigma"]
    assert abs(chosen["trend_ctmle_limit_bias"]) >= pilot.CTMLE_LIMIT_FLOOR * chosen["trend_sigma"]
    assert chosen["trend_both_wrong_pass_probability"] >= pilot.PASS_PROBABILITY
    assert chosen["trend_outcome_correct_pass_probability"] >= pilot.PASS_PROBABILITY


@pytest.mark.parametrize("cell", ["both_wrong", "outcome_correct"])
def test_the_committed_pilot_rows_reproduce(cell: str) -> None:
    """Replicate 0 of each pilot cell at the chosen coefficient refits to the committed bytes."""
    rows = pd.read_csv(pilot.PILOT_REPLICATE_ROWS, float_precision="round_trip")
    curvature = laws.OUTCOME_CURVATURE
    committed = rows.loc[
        (rows["curvature"] == curvature) & (rows["cell"] == cell) & (rows["replicate"] == 0)
    ].iloc[0]
    refit = pilot._pilot_fit((curvature, cell, 0))
    for column in ("estimate", "std_error", "tmle_estimate"):
        assert refit[column] == pytest.approx(committed[column], rel=1e-9, abs=1e-12)


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


def test_the_declared_pair_is_the_lower_median_design_draw_and_reproduces() -> None:
    """``penalties.csv`` is the median draw's pair, and refitting that draw returns it.

    The refit agrees to a relative ``1e-12``: the cross-validated glmnet path moves in the last
    bit between platforms and Python builds (seen on the CI Python 3.12 job).
    """
    design = pd.read_csv(study.PENALTY_DESIGN, float_precision="round_trip")
    assert len(design) == study.PENALTY_DRAWS
    assert list(design["seed"]) == [study.PENALTY_SEED + draw for draw in design["draw"]]
    chosen = study.median_draw(design)
    assert study.declared_pair() == (float(chosen["lambda_cv"]), float(chosen["partner"]))
    mechanism, partner = study.declared_pair()
    assert 0.0 < partner < mechanism
    refit = study._penalty_draw(int(chosen["draw"]))
    assert (refit["lambda_cv"], refit["partner"]) == pytest.approx((mechanism, partner), rel=1e-12)


def test_the_cells_carry_the_declared_budgets() -> None:
    budgets = {(cell.family, cell.cell): (cell.n, cell.replicates) for cell in properties.cells()}
    calibration = (properties.CALIBRATION_N, properties.CALIBRATION_REPLICATES)
    for label in study.PROPERTY_CELLS["interval_calibration"]:
        assert budgets["interval_calibration", label] == calibration
    assert budgets["type_i_error", "sharp_null"] == (properties.CALIBRATION_N, 2_400)
    assert budgets["power", "alternative"] == (laws.POWER_N, laws.POWER_REPLICATES)
    for n in properties.LADDER:
        assert budgets["root_n_and_efficiency", f"n_{n}"] == (n, properties.LADDER_REPLICATES)
    for label in study.PROPERTY_CELLS["double_robustness"]:
        expected = 1_800 if label == "multi_arm_treatment_correct" else 1_200
        assert budgets["double_robustness", label] == (2_000, expected)
    assert budgets["targeting_necessity", "ate__targeted"] == (2_000, 1_200)
    assert budgets["clustered_inference", "cluster_robust"] == calibration
    declared = {(cell.property, cell.cell) for cell in properties.declared_cells()}
    published = {
        (family, label)
        for family, labels in study.PROPERTY_CELLS.items()
        if family != "root_n_rate"
        for label in labels
    }
    assert declared == published


def test_the_targeting_threshold_sits_below_the_exact_untargeted_displacement() -> None:
    """The GLM without ``W2`` has a limit bias, and targeting must remove it.

    The limit fits the wrong GLM on ``4 x 10^5`` draws.  The targeted arm's spread is the
    influence-curve SD over root n, with that wrong outcome fit and the true mechanism.  The
    declared threshold must sit well below the resulting displacement, so a working targeting
    step clears it and an inert one cannot.
    """
    law = laws.PRIMARY
    rng = np.random.default_rng(20263006)
    w = rng.standard_normal((400_000, 4))
    g = np.clip(law.mechanism(w)[:, 1], *laws.G_BOUNDS)
    a = (rng.uniform(size=g.size) < g).astype(float)
    y = (
        rng.uniform(size=g.size)
        < np.where(a == 1.0, law.outcome_probability(1, w), law.outcome_probability(0, w))
    ).astype(float)
    design = np.column_stack([a, w])
    fit = study.omitted_confounder_learner().fit(design, y)

    def predict(arm: float) -> np.ndarray:
        return fit.predict_proba(np.column_stack([np.full(g.size, arm), w]))[:, 1]

    q0, q1 = predict(0.0), predict(1.0)
    truth = law.truth()["ate"]
    bias = float(np.mean(q1 - q0)) - truth
    curve = (a / g - (1.0 - a) / (1.0 - g)) * (y - np.where(a == 1.0, q1, q0)) + q1 - q0 - truth
    spread = curve.std() / np.sqrt(properties.NECESSITY_N)
    assert abs(bias) / spread >= 1.5 * properties.TARGETING_DISPLACEMENT


def test_the_clustered_law_has_a_large_cluster_design_effect() -> None:
    """The oracle influence curve's cluster design effect clears the declared floor.

    At ``10^4`` clusters the estimate's relative error is about 1.5%.  The IID control's SE
    ratio is about one over the root of the design effect, so the floor keeps it below the
    family's ceiling.
    """
    law = laws.CLUSTERED
    assert law.clusters is not None
    frame = law.sample(law.clusters * 10_000, 20263007)
    w = frame[list(laws.COVARIATES)].to_numpy()
    g = law.mechanism(w)[:, 1]
    a = frame["A"].to_numpy()
    q0, q1 = law.outcome_probability(0, w), law.outcome_probability(1, w)
    residual = frame["Y"].to_numpy() - np.where(a == 1.0, q1, q0)
    curve = (a / g - (1.0 - a) / (1.0 - g)) * residual + q1 - q0 - law.truth()["ate"]
    sums = np.bincount(frame["cluster"].to_numpy(), weights=curve)
    effect = float(sums.var() / law.clusters / curve.var())
    assert effect >= laws.CLUSTER_DESIGN_EFFECT
    assert 1.0 / np.sqrt(laws.CLUSTER_DESIGN_EFFECT) < CLUSTER_ROBUST_CONTROL_SE_CEILING


@pytest.mark.parametrize(
    ("law", "estimand", "declared", "error"),
    [
        (laws.PRIMARY, "ate", laws.PRIMARY_EIF_SD, laws.PRIMARY_EIF_SD_ERROR),
        (laws.THREE_ARM, "ate[1.0 vs 0.0]", laws.THREE_ARM_EIF_SD, laws.THREE_ARM_EIF_SD_ERROR),
        (
            laws.MISSING_OUTCOME,
            "ate",
            laws.MISSING_OUTCOME_EIF_SD,
            laws.MISSING_OUTCOME_EIF_SD_ERROR,
        ),
    ],
    ids=["primary", "three_arm", "missing_outcome"],
)
def test_each_efficiency_bound_reproduces(
    law: laws.DerivativeLaw, estimand: str, declared: float, error: float
) -> None:
    """``10^5`` draws reproduce each ``10^6``-draw bound to four of their own errors."""
    sd, own = law.eif_sd(100_000, 2, estimand)
    assert abs(sd - declared) < 4.0 * max(own, error)
    assert properties.EFFICIENCY_BOUNDS


def _primary_rows() -> pd.DataFrame:
    return pd.read_csv(study.ARTIFACTS / "replicates.csv.gz", float_precision="round_trip")


def test_the_parity_witness_reproduces_and_passes() -> None:
    """On each parity scenario, Python and R differ far less than the derivative moves.

    ``paired-tmle.csv`` publishes the SD of the C-TMLE0 minus TMLE difference.  The SD of
    the Python minus R difference, recomputed here from the committed rows, must be at most
    :data:`~tests.studies.canonical_ctmle_derivative.PARITY_WITNESS_RATIO` of it.
    """
    rows = _primary_rows()
    paired = pd.read_csv(study.ARTIFACTS / "paired-tmle.csv", float_precision="round_trip")
    if study.STUDY.comparator_state == "pending":
        # The R rows land with the comparator.  Until then no row and no verdict may appear.
        assert not (rows["implementation"] == study.REFERENCE).any()
        assert paired["sd_parity_difference"].isna().all()
        assert paired["parity_witness_passed"].isna().all()
        return
    for scenario in study.PARITY_SCENARIOS:
        frame = rows.loc[rows["scenario"] == scenario].pivot(
            index="replicate", columns="implementation", values="estimate"
        )
        parity = float((frame[study.IMPLEMENTATION] - frame[study.REFERENCE]).std(ddof=1))
        row = paired.loc[paired["scenario"] == scenario].iloc[0]
        assert parity == pytest.approx(row["sd_parity_difference"], rel=1e-9)
        assert parity <= study.PARITY_WITNESS_RATIO * row["sd_difference"]
        assert bool(row["parity_witness_passed"])


def test_a_side_without_the_derivative_columns_would_fail_the_witness() -> None:
    """The deliberate-mutation control of the witness, on five ``binary_constant_q`` draws.

    Each refit matches R to half the bound.  The same fits retargeted without the derivative
    columns, the plain TMLE, sit farther from R in root mean square than the witness allows a
    scenario's SD to be.
    """
    if study.STUDY.comparator_state == "pending":
        pytest.skip("the control reads R rows, which land with the pending comparator")
    scenario = "binary_constant_q"
    rows = _primary_rows()
    paired = pd.read_csv(study.ARTIFACTS / "paired-tmle.csv", float_precision="round_trip")
    bound = study.PARITY_WITNESS_RATIO * float(
        paired.loc[paired["scenario"] == scenario, "sd_difference"].iloc[0]
    )
    plain = []
    for replicate in range(5):
        frame, _ = study.draw_scenario(scenario, study.PRIMARY_N, replicate)
        result = study.fit_cleverly(frame, scenario)
        reference = float(
            rows.loc[
                (rows["scenario"] == scenario)
                & (rows["replicate"] == replicate)
                & (rows["implementation"] == study.REFERENCE),
                "estimate",
            ].iloc[0]
        )
        # Python bounds each targeted arm prediction at 0.005 and 0.995 after every step, and
        # R bounds only the initial fit.  Under a constant outcome the derivative coefficients
        # are large, so the two differ by up to about 1e-4, about a fifth of the bound.
        assert abs(float(result["ate"].psi) - reference) < 0.5 * bound
        plain.append(float(study.paired_tmle(result)["ate"].psi) - reference)
    assert float(np.sqrt(np.mean(np.square(plain)))) > bound


def test_the_cluster_robust_cell_is_sized_for_its_pass_probability() -> None:
    """The cluster cell passes with probability 0.9 or more at a true SE ratio of 1.

    With 100 clusters the reported standard error varies more than with independent rows.  A
    200-replicate pilot measured a relative spread of 0.075, so the sizing carries it.
    """
    margins = Margins()
    probability = calibration_pass_probability(
        properties.CALIBRATION_REPLICATES,
        true_se_ratio=1.0,
        se_noise=0.075,
        confidence_level=margins.confidence_level,
        se_band=margins.calibration_se_ratio,
        coverage_band=margins.calibration_coverage,
    )
    assert probability >= 0.90
