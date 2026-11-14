"""The design of ``few-cluster-adaptive-learners``: its laws, cells, rules, controls and pilot.

The L2 truth is exact quadrature (:mod:`tests.studies.clustered_nonlinear_law`) and is checked
against a large draw of the same law, so a wrong formula cannot pass by agreeing with itself.
The declared rules (:mod:`tests.studies.few_cluster_adaptive_properties`) are checked on
constructed rows and summaries, each against a mutation that must change the verdict. The
pilot (:mod:`tests.studies.few_cluster_adaptive_pilot`) is read from its committed rows, and the
predicted pass probabilities the declaration states are recomputed from them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from tests.studies import clustered_few_cluster_properties as committed_few_cluster
from tests.studies import clustered_nonlinear_law as nonlinear
from tests.studies import few_cluster_adaptive_learners as study
from tests.studies import few_cluster_adaptive_pilot as pilot
from tests.studies import few_cluster_adaptive_properties as properties
from tests.studies.evidence.registry import registered

PILOT = study.STUDY.artifacts / "pilot.csv.gz"


@pytest.fixture(scope="module")
def pilot_rows() -> pd.DataFrame:
    return pd.read_csv(PILOT, keep_default_na=False, na_values=[""])


@pytest.fixture(scope="module")
def readings(pilot_rows: pd.DataFrame) -> pd.DataFrame:
    return pilot.pilot_readings(pilot_rows).set_index("cell")


# ------------------------------------------------------------------------------ the record


def test_the_record_publishes_every_declared_cell() -> None:
    declared = {cell.cell for cell in properties.declared_cells()}
    assert set(study.STUDY.property_cells[study.FAMILY]) == declared == set(study.PUBLISHED_CELLS)
    assert len(declared) == 64
    roles = {cell.cell: cell.role for cell in properties.declared_cells()}
    positives = {name for name, role in roles.items() if role == "positive"}
    assert positives == {name for name in declared if name.endswith("__t_reference")}
    # 2 point laws x 3 counts x 2 default-library fits, and the cross-fitted LTMLE cell.
    assert len(positives) == 13
    controls = {name for name, role in roles.items() if role == "control"}
    assert len(controls) == 13 + 12
    for name in positives:
        assert roles[properties.iid_cell(name)] == "control"
        assert roles[properties.must_fail_cell(name)] == "control"
        assert properties.must_fail_cell(name).split("__")[-1] in {
            "interpolating_control",
            "stacked_control",
        }


def test_the_seeds_are_the_study_s_own() -> None:
    seeds = {s for record in registered() for s in (record.seed, record.resampling_seed)}
    own = {study.SEED, study.RESAMPLING_SEED, pilot.PILOT_SEED}
    assert not own & seeds
    assert len(own) == 3


def test_declared_and_not_yet_registered() -> None:
    from tests.unit.test_method_evidence import UNCOMMITTED_MODULES

    assert study.STUDY.slug not in {record.slug for record in registered()}
    assert not study.STUDY.artifact("property-replicates.parquet").exists()
    assert properties.__name__ in UNCOMMITTED_MODULES
    assert study.STUDY.publication_policy == "reporting"
    assert properties.FAILURE_BOUND == 0.05
    assert study.STUDY.failure_bound == {"cleverly": 0.05}
    assert study.PROPERTY_REPLICATES == 4_000


def test_the_declared_thresholds() -> None:
    assert pytest.approx(0.8649) == properties.VARIANCE_RATIO_FLOOR
    assert properties.SHOWN_COVERAGE_FLOOR == 0.92
    assert properties.NONINFERIORITY_MARGIN == -0.02
    assert properties.MUST_FAIL_CEILING == 0.90


# --------------------------------------------------------------------------------- the laws


def test_the_l1_truth_is_the_committed_equal10_truth() -> None:
    committed = pd.read_parquet(
        Path(__file__).resolve().parents[1]
        / "canonical"
        / "ltmle_few_cluster"
        / "property-replicates.parquet",
        columns=["cell", "truth"],
    )
    values = committed.loc[committed["cell"].str.contains("__equal10__"), "truth"].unique()
    assert values.size == 1
    assert study.law_truth("equal10") == pytest.approx(float(values[0]), abs=1e-6)
    assert study.law_truth("equal10") == committed_few_cluster.SizeLaw("equal10/J20").truth()["ate"]


def test_the_l2_truth_agrees_across_resolutions() -> None:
    declared = nonlinear.truth(nonlinear.RESOLUTION)
    check = nonlinear.truth(nonlinear.CHECK_RESOLUTION)
    for key in ("ey0", "ey1", "ate"):
        assert abs(declared[key] - check[key]) < nonlinear.TRUTH_AGREEMENT


def test_the_l2_truth_matches_a_large_draw() -> None:
    """Witness: the counterfactual means of 400,000 draws, at 4.5 Monte Carlo SEs."""
    rng = np.random.default_rng(31)
    size = 400_000
    w1, w2, u = (rng.normal(size=size) for _ in range(3))
    truth = nonlinear.truth()
    for arm in (0, 1):
        mean = expit(nonlinear.outcome_logit(w1, w2, arm) + nonlinear.latent_scale(arm) * u)
        error = float(mean.std() / np.sqrt(size))
        assert abs(float(mean.mean()) - truth[f"ey{arm}"]) < 4.5 * error


def test_a_wrong_latent_scale_moves_the_l2_truth() -> None:
    """Mutation: the arm-by-latent term moves the truth by far more than the agreement."""
    points = nonlinear.RESOLUTION[1]
    x, w = np.polynomial.hermite_e.hermegauss(40)
    w = w / np.sqrt(2.0 * np.pi)
    w1, w2 = (grid.ravel() for grid in np.meshgrid(x, x, indexing="ij"))
    weight = np.outer(w, w).ravel()
    center = nonlinear.outcome_logit(w1, w2, 1)
    declared = weight @ nonlinear.logistic_normal(center, float(nonlinear.latent_scale(1)), points)
    without = weight @ nonlinear.logistic_normal(center, 0.6, points)
    assert abs(declared - without) > 0.01


def test_l2_misspecifies_a_main_terms_glm_for_both_nuisances() -> None:
    """Witness: the curvature terms cut the deviance of each nuisance by a chi-square margin."""
    frame = nonlinear.draw(2_000, np.random.default_rng(3))
    w1, w2 = frame["W1"].to_numpy(), frame["W2"].to_numpy()
    main = np.column_stack([w1, w2])

    def deviance(design: np.ndarray, target: np.ndarray) -> float:
        model = LogisticRegression(C=1e8, max_iter=5_000).fit(design, target)
        p = np.clip(model.predict_proba(design)[:, 1], 1e-12, 1 - 1e-12)
        return float(-2 * np.sum(target * np.log(p) + (1 - target) * np.log(1 - p)))

    a, y = frame["A"].to_numpy(), frame["Y"].to_numpy()
    threshold = float(stats.chi2.ppf(0.999, 2))
    g_gain = deviance(main, a) - deviance(np.column_stack([main, w1**2]), a)
    assert g_gain > threshold
    outcome = np.column_stack([a, main])
    q_gain = deviance(outcome, y) - deviance(np.column_stack([outcome, np.sin(2 * w1), w1 * w2]), y)
    assert q_gain > threshold


def test_the_l2_propensity_stays_inside_its_stated_range() -> None:
    w1, w2 = np.meshgrid(np.linspace(-3, 3, 301), np.linspace(-2, 2, 201))
    g = nonlinear.propensity(w1, w2)
    assert 0.017 <= g.min() < 0.02
    assert 0.83 < g.max() <= 0.84


# ----------------------------------------------------------------------------- the controls


def test_the_full_sample_tree_predicts_seen_outcomes_on_validation_rows() -> None:
    """Witness: a fold's out-of-fold predictions are the full-sample tree's in-sample fit."""
    frame = study.draw("equal10", 20, 5)
    design = frame[["A", "W1", "W2"]].to_numpy()
    target = frame["Y"].to_numpy()
    tree = study.FullSampleTree(design, target).fit(design[:150], target[:150])
    held = tree.predict_proba(design[150:])[:, 1]
    np.testing.assert_array_equal(held, target[150:])


def test_the_full_sample_tree_refuses_a_design_that_is_not_its_own() -> None:
    """Mutation: a changed column order stops the control instead of fitting another design."""
    frame = study.draw("equal10", 20, 5)
    design = frame[["A", "W1", "W2"]].to_numpy()
    target = frame["Y"].to_numpy()
    with pytest.raises(ValueError, match="not a row of the full design"):
        study.FullSampleTree(design, target).fit(design[:, [1, 0, 2]], target)


def test_the_stacked_control_feeds_the_tree_into_the_fit() -> None:
    estimator = study.point_estimator("stacked_control", study.draw("equal10", 20, 5))
    assert isinstance(estimator.outcome_learner, study.FullSampleTree)
    assert estimator.cross_fit


# --------------------------------------------------------------------------------- the rows


class _Withheld:
    """A real estimate whose interval reads as withheld, every other attribute its own."""

    def __init__(self, estimate: object) -> None:
        self._estimate = estimate

    supplies_inference = False

    def __getattr__(self, name: str) -> object:
        return getattr(self._estimate, name)


@pytest.fixture(scope="module")
def glm_fit() -> study.Fitted:
    return study.fit_one("glm_in_sample", study.draw("equal10", 20, 2))


def test_a_withheld_fit_is_a_miss_in_every_arm(glm_fit: study.Fitted) -> None:
    """Mutation control: the same fit shown covers in every arm, withheld in none."""
    assert glm_fit.estimate.supplies_inference
    shown = properties.fit_rows("tmle_in_sample", "equal10", 20, 0, glm_fit, 10)
    withheld = properties.fit_rows(
        "tmle_in_sample", "equal10", 20, 0, study.Fitted(_Withheld(glm_fit.estimate), 1.0), 10
    )
    assert [row["covered"] for row in shown] == [1, 1, 1]
    assert [row["covered"] for row in withheld] == [0, 0, 0]
    assert {row["withheld"] for row in withheld} == {1}
    assert {row["withheld"] for row in shown} == {0}


def test_the_wang_arm_widens_the_reported_interval(glm_fit: study.Fitted) -> None:
    rows = properties.fit_rows("tmle_in_sample", "equal10", 20, 0, glm_fit, 10)
    by_arm = {row["cell"].split("__")[-1]: row for row in rows}
    plugin = float(glm_fit.estimate.plugin_std_error)
    assert by_arm["wang_inflated"]["std_error"] == pytest.approx(plugin * np.sqrt(20 / 18))
    assert by_arm["t_reference"]["std_error"] == pytest.approx(float(glm_fit.estimate.std_error))


# -------------------------------------------------------------------------------- the rules


def _passing_summary() -> pd.DataFrame:
    """A summary on which every declared verdict passes."""
    rows = []
    for cell in properties.declared_cells():
        rows.append(
            {
                "property": study.FAMILY,
                "cell": cell.cell,
                "role": cell.role,
                "coverage_ci_lower": 0.93,
                "coverage_ci_upper": 0.96,
                "bias_equivalent": True,
                "variance_ratio_ci_lower": 0.95,
                "shown_ci_lower": 0.93,
                "noninferiority_lower": 0.0,
                "paired_se_ratio_ci_upper": 0.6,
                "paired_positive_se_ratio_ci_upper": 1.1,
            }
        )
    frame = pd.DataFrame(rows)
    controls = frame["cell"].str.endswith(("interpolating_control", "stacked_control"))
    frame.loc[controls, ["coverage_ci_lower", "coverage_ci_upper"]] = (0.6, 0.7)
    return frame


POSITIVE = study.cell_name("tmle_in_sample", "equal10", 20, "t_reference")
NONLINEAR = study.cell_name("tmle_in_sample", "nonlinear10", 20, "t_reference")


def _verdict(summary: pd.DataFrame, cell: str, column: str = "passed") -> bool:
    properties.adaptive_verdicts(summary)
    return bool(summary.loc[summary["cell"] == cell, column].iloc[0])


def test_every_constructed_verdict_passes() -> None:
    summary = _passing_summary()
    properties.adaptive_verdicts(summary)
    assert summary["passed"].all()
    positive = summary["role"] == "positive"
    assert summary.loc[positive, "caveat_removed"].eq(1.0).all()


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("coverage_ci_lower", 0.89),
        ("bias_equivalent", False),
        ("variance_ratio_ci_lower", 0.86),
        ("shown_ci_lower", 0.91),
        ("noninferiority_lower", -0.021),
    ],
)
def test_each_positive_clause_fails_its_row(column: str, value: object) -> None:
    summary = _passing_summary()
    summary.loc[summary["cell"] == POSITIVE, column] = value
    assert not _verdict(summary, POSITIVE)
    assert not _verdict(summary, POSITIVE, "caveat_removed")


def test_the_variance_ratio_clause_is_one_sided() -> None:
    """An over-reported variance passes the declared clause: only the lower end is read."""
    summary = _passing_summary()
    summary.loc[summary["cell"] == POSITIVE, "variance_ratio_ci_lower"] = 1.6
    assert _verdict(summary, POSITIVE)


def test_non_inferiority_reads_l1_only() -> None:
    summary = _passing_summary()
    summary.loc[summary["cell"] == NONLINEAR, "noninferiority_lower"] = -0.5
    assert _verdict(summary, NONLINEAR)


def test_a_covering_must_fail_control_keeps_the_caveat() -> None:
    summary = _passing_summary()
    control = properties.must_fail_cell(POSITIVE)
    summary.loc[summary["cell"] == control, ["coverage_ci_lower", "coverage_ci_upper"]] = (
        0.88,
        0.905,
    )
    assert not _verdict(summary, control)
    assert _verdict(summary, POSITIVE)
    assert not _verdict(summary, POSITIVE, "caveat_removed")


def test_a_failed_iid_control_keeps_the_caveat() -> None:
    summary = _passing_summary()
    summary.loc[summary["cell"] == properties.iid_cell(POSITIVE), "paired_se_ratio_ci_upper"] = 0.81
    assert _verdict(summary, POSITIVE)
    assert not _verdict(summary, POSITIVE, "caveat_removed")


def test_the_ltmle_cell_reads_the_stacked_control_of_l1_at_20() -> None:
    cell = study.cell_name("ltmle_crossfit", "equal40", 20, "t_reference")
    assert properties.must_fail_cell(cell) == study.cell_name(
        "stacked_control", "equal10", 20, "stacked_control"
    )


def _paired(covered: list[int], withheld: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {"replicate": range(len(covered)), "covered": covered, "withheld": withheld}
    )


def test_non_inferiority_counts_a_withheld_draw_against_the_default_library() -> None:
    """Mutation: the same pairs read on shown draws only hide the withheld misses."""
    # 30 draws where the default library withholds (a miss) and the GLM covers.
    positive = _paired([1] * 470 + [0] * 30, [0] * 470 + [1] * 30)
    glm = _paired([1] * 500, [0] * 500)
    counts = properties.paired_counts(positive, glm)
    assert counts == {"trials": 500, "gained": 0, "lost": 30}
    lower = properties.mcnemar_lower(counts["gained"], counts["lost"], counts["trials"])
    assert lower <= properties.NONINFERIORITY_MARGIN
    shown = positive.loc[positive["withheld"] == 0]
    mutated = properties.paired_counts(shown, glm)
    assert properties.mcnemar_lower(mutated["gained"], mutated["lost"], mutated["trials"]) == 0.0


def test_a_lost_fit_reads_against_the_default_library() -> None:
    positive = _paired([1] * 9, [0] * 9)
    glm = _paired([1] * 10, [0] * 10)
    assert properties.lost_counts(positive, glm, 10) == {"trials": 10, "gained": 0, "lost": 1}
    assert properties.lost_counts(glm, positive, 10) == {"trials": 10, "gained": 0, "lost": 0}


def test_shown_coverage_sees_a_variance_ratio_carried_by_outliers() -> None:
    """Witness of plan N3: a few huge SEs lift the ratio over its floor while coverage fails."""
    rng = np.random.default_rng(4)
    errors = rng.normal(size=4_000)
    se = np.full(4_000, 0.80)
    se[:40] = 9.0  # one draw in a hundred reports a huge interval
    ratio, low, _ = properties.variance_ratio_interval(
        errors, se, replicates=2_000, confidence_level=0.99, seed=1
    )
    assert low >= properties.VARIANCE_RATIO_FLOOR
    covered = np.abs(errors) <= 1.96 * se
    shown_low = stats.beta.ppf(0.005, covered.sum(), covered.size - covered.sum() + 1)
    assert shown_low < properties.SHOWN_COVERAGE_FLOOR
    assert ratio > 1.0


def test_the_variance_ratio_interval_sees_a_short_standard_error() -> None:
    """Mutation: SEs at 0.9 of the spread put the 99% upper end below the floor at 4,000 draws."""
    rng = np.random.default_rng(6)
    errors = rng.normal(size=4_000)
    _, _, high = properties.variance_ratio_interval(
        errors, np.full(4_000, 0.88), replicates=2_000, confidence_level=0.99, seed=2
    )
    assert high < properties.VARIANCE_RATIO_FLOOR


# -------------------------------------------------------------------------------- the pilot


def test_the_pilot_holds_every_declared_key(pilot_rows: pd.DataFrame) -> None:
    keys = pilot.pilot_keys()
    expected = sum(len(study.ARMS[fit]) for *_, fit in keys)
    assert len(pilot_rows) == expected
    assert set(pilot_rows["cell"]) == set(study.PUBLISHED_CELLS)


def test_the_predicted_pass_probabilities_are_the_declared_ones(readings: pd.DataFrame) -> None:
    for cell, declared in study.PREDICTED_PASS.items():
        assert readings.loc[cell, "p_passed"] == pytest.approx(declared, abs=0.01), cell


def test_every_predicted_red_is_declared(readings: pd.DataFrame) -> None:
    """A gated row predicted below 0.10 is named in the declaration's predicted reds."""
    gated = readings.loc[readings["role"] != "diagnostic"]
    reds = set(gated.index[gated["p_passed"] < 0.10])
    assert reds == set(study.PREDICTED_REDS)


def test_the_pass_probabilities_see_a_shortfall() -> None:
    """Mutation controls of the predictive probabilities."""
    assert pilot.lower_end_pass(58, 60, 4_000, 0.90) > 0.9
    assert pilot.lower_end_pass(52, 60, 4_000, 0.90) < 0.3
    assert pilot.upper_end_fails(30, 60, 4_000, 0.90) > 0.99
    assert pilot.upper_end_fails(57, 60, 4_000, 0.90) < 0.1
    assert pilot.normal_pass(0.0, 0.05, 0.01, -0.13) > 0.9
    assert pilot.normal_pass(-0.2, 0.05, 0.01, -0.13) < 0.1
