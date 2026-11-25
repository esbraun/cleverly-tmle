"""The design of ``few-cluster-adaptive-learners``: its laws, cells, rules, controls and pilot.

The L2 truth is exact quadrature (:mod:`tests.studies.clustered_nonlinear_law`) and is checked
against a large draw of the same law, so a wrong formula cannot pass by agreeing with itself.
The declared rules (:mod:`tests.studies.few_cluster_adaptive_properties`) are checked on
constructed rows and summaries, each against a mutation that must change the verdict. The
pilot (:mod:`tests.studies.few_cluster_adaptive_pilot`) is read from its committed rows, and the
predicted pass probabilities the declaration states are recomputed from them.
"""

from __future__ import annotations

import json
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


# ------------------------------------------------------------------------------ the record


def test_the_record_publishes_every_declared_cell() -> None:
    declared = {cell.cell for cell in properties.declared_cells()}
    assert set(study.STUDY.property_cells[study.FAMILY]) == declared == set(study.PUBLISHED_CELLS)
    assert len(declared) == 64
    roles = {cell.cell: cell.role for cell in properties.declared_cells()}
    positives = {name for name, role in roles.items() if role == "positive"}
    # 2 point laws x 3 counts of stacked TMLE, and the cross-fitted LTMLE cell.
    assert positives == {
        name
        for name in declared
        if name.endswith("__t_reference") and name.split("__")[0] in study.POSITIVE_FITS
    }
    assert len(positives) == 7
    controls = {name for name, role in roles.items() if role == "control"}
    assert controls == {properties.iid_cell(name) for name in positives} | {
        name for name in declared if name.endswith("__stacked_control")
    }
    assert len(controls) == 7 + 6
    for name in positives:
        assert properties.must_fail_cell(name).endswith("__stacked_control")
    # The in-sample default-library fit and the interpolating control are reported only.
    for name in declared:
        if name.split("__")[0] in {"tmle_in_sample", "interpolating_control"}:
            assert roles[name] == "diagnostic", name


def test_the_seeds_are_the_study_s_own() -> None:
    seeds = {
        s
        for record in registered()
        if record.slug != study.STUDY.slug
        for s in (record.seed, record.resampling_seed)
    }
    own = {study.SEED, study.RESAMPLING_SEED, pilot.PILOT_SEED}
    assert not own & seeds
    assert len(own) == 3


def test_registered_with_its_declared_run() -> None:
    from tests.unit.test_method_evidence import UNCOMMITTED_MODULES

    assert study.STUDY.slug in {record.slug for record in registered()}
    assert study.STUDY.artifact("property-replicates.parquet").exists()
    assert properties.__name__ not in UNCOMMITTED_MODULES
    assert study.STUDY.publication_policy == "reporting"
    assert properties.FAILURE_BOUND == 0.01
    assert study.STUDY.failure_bound == {"cleverly": 0.01}
    assert study.PROPERTY_REPLICATES == 4_000
    assert study.STUDY.replicates == study.PRIMARY_REPLICATES == 1_200
    assert study.INTERIM_DRAWS is None


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


def _control_inputs() -> tuple[np.ndarray, np.ndarray]:
    frame = study.draw("equal10", 20, 5)
    return frame[["A", "W1", "W2"]].to_numpy(), frame["Y"].to_numpy()


def test_the_full_sample_forest_predicts_validation_rows_from_the_full_sample() -> None:
    """Witness: a fold's out-of-fold predictions are the full-sample forest's in-sample fit."""
    design, target = _control_inputs()
    folded = study.FullSampleForest(design, target).fit(design[:150], target[:150])
    full = study.FullSampleForest(design, target).fit(design, target)
    held = folded.predict_proba(design[150:])[:, 1]
    np.testing.assert_allclose(held, full.predict_proba(design[150:])[:, 1])


def test_the_stacked_control_shrinks_the_residuals_without_removing_them() -> None:
    """The residuals shrink well below an honest fit's but do not vanish.

    An interpolating tree leaves every residual at 0, so the fluctuation step and the
    propensity drop out and the stacked control repeats the in-sample one. The forest's
    residuals stay away from 0, so both still act.
    """
    design, target = _control_inputs()
    forest = study.FullSampleForest(design, target).fit(design, target)
    residual = np.abs(target - forest.predict_proba(design)[:, 1])
    honest = np.abs(target - target.mean())
    assert residual.mean() < 0.5 * honest.mean()
    assert np.mean(residual > 0.01) > 0.5


def test_the_full_sample_forest_refuses_a_design_that_is_not_its_own() -> None:
    """Mutation: a changed column order stops the control instead of fitting another design."""
    design, target = _control_inputs()
    with pytest.raises(ValueError, match="not a row of the full design"):
        study.FullSampleForest(design, target).fit(design[:, [1, 0, 2]], target)


def test_the_two_controls_differ_on_a_draw() -> None:
    """The stacked control no longer repeats the interpolating one."""
    frame = study.draw("equal10", 20, 7)
    stacked = study.fit_one("stacked_control", frame).estimate
    interpolating = study.fit_one("interpolating_control", frame).estimate
    assert abs(float(stacked.psi) - float(interpolating.psi)) > 1e-3
    assert abs(float(stacked.std_error) - float(interpolating.std_error)) > 1e-3


def test_the_stacked_control_feeds_the_forest_into_the_fit() -> None:
    estimator = study.point_estimator("stacked_control", study.draw("equal10", 20, 5))
    assert isinstance(estimator.outcome_learner, study.FullSampleForest)
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


def test_every_row_carries_the_declared_role_of_its_arm(glm_fit: study.Fitted) -> None:
    """The rows of a reported fit carry its diagnostic roles, and of a gated fit its own."""
    for fit in ("tmle_in_sample", "tmle_stacked"):
        rows = properties.fit_rows(fit, "equal10", 20, 0, glm_fit, 10)
        assert [row["role"] for row in rows] == [role for _, role in study.ARMS[fit]]


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
    # A reported in-sample row reads like a failing one: it must gate nothing.
    frame.loc[frame["cell"] == REPORTED, ["coverage_ci_lower", "variance_ratio_ci_lower"]] = (
        0.5,
        0.3,
    )
    frame.loc[controls, ["coverage_ci_lower", "coverage_ci_upper"]] = (0.6, 0.7)
    return frame


POSITIVE = study.cell_name("tmle_stacked", "equal10", 20, "t_reference")
NONLINEAR = study.cell_name("tmle_stacked", "nonlinear10", 20, "t_reference")
REPORTED = study.cell_name("tmle_in_sample", "equal10", 20, "t_reference")


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


def test_a_reported_in_sample_row_gates_nothing() -> None:
    """The in-sample row reads red values and still states no verdict."""
    summary = _passing_summary()
    properties.adaptive_verdicts(summary)
    reported = summary.loc[summary["cell"] == REPORTED].iloc[0]
    assert reported["role"] == "diagnostic"
    assert bool(reported["passed"])
    assert pd.isna(reported.get("caveat_removed", np.nan))
    assert not properties.noninferiority_gated(REPORTED)
    assert properties.noninferiority_gated(POSITIVE)


def test_the_coverage_floor_is_implied_only_at_a_small_withheld_share() -> None:
    """The binding reason holds while few intervals are withheld.

    At 4,000 draws and shown coverage 0.95 the shown lower end clears 0.92. With no interval
    withheld the all-draw lower end clears 0.90 too; with 5% withheld it does not, so there the
    floor decides alone and the differential's reason check would be refuted.
    """
    draws, coverage = 4_000, 0.95

    def lower(hits: float, trials: float) -> float:
        return float(stats.beta.ppf(0.005, hits, trials - hits + 1))

    for withheld, implied in ((0.0, True), (0.05, False)):
        shown = round(draws * (1 - withheld))
        hits = round(shown * coverage)
        assert lower(hits, shown) >= properties.SHOWN_COVERAGE_FLOOR
        assert (lower(hits, draws) >= 0.90) is implied


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


def test_the_pilot_runs_every_fit_of_every_cell_at_400_draws() -> None:
    keys = pilot.pilot_keys()
    assert pilot.PILOT_DRAWS == 400
    cells = {(law, clusters, fit) for law, clusters, _, fit in keys}
    assert cells == {
        (law, clusters, fit) for law, clusters in study.DRAW_CELLS for fit in study.fits_of(law)
    }
    assert len(keys) == len(cells) * 400 == 38 * 400
    chunks = pilot.pilot_chunks()
    assert len(chunks) == len(set(chunks)) == len(cells) * 400 // pilot.CHUNK_DRAWS
    assert chunks[0][2] in {"tmle_stacked", "ltmle_crossfit"}


def test_the_pilot_resumes_from_its_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A written chunk is never refitted, and a missing one is."""
    monkeypatch.setattr(pilot, "PILOT_DRAWS", 2)
    monkeypatch.setattr(pilot, "CHUNK_DRAWS", 1)
    fitted: list[tuple[str, int, int, str]] = []

    def fake(key: tuple[str, int, int, str]) -> list[dict[str, object]]:
        fitted.append(key)
        law, clusters, replicate, fit = key
        return [
            {"cell": study.cell_name(fit, law, clusters, arm), "replicate": replicate, "lost": ""}
            for arm, _ in study.ARMS[fit]
        ]

    monkeypatch.setattr(pilot, "key_rows", fake)
    chunks = pilot.pilot_chunks()
    (tmp_path / "chunks").mkdir()
    for chunk in chunks[1:]:
        pilot.run_chunk(tmp_path, chunk)
    fitted.clear()
    todo = [c for c in pilot.pilot_chunks() if not pilot.chunk_path(tmp_path, c).exists()]
    assert todo == [chunks[0]]
    pilot.run_chunk(tmp_path, todo[0])
    assert len(fitted) == 1
    assert all(pilot.chunk_path(tmp_path, c).exists() for c in chunks)


def test_the_committed_pilot_holds_every_declared_key(pilot_rows: pd.DataFrame) -> None:
    expected = sum(len(study.ARMS[fit]) for *_, fit in pilot.pilot_keys())
    assert len(pilot_rows) == expected == 25_600
    assert set(pilot_rows["cell"]) == set(study.PUBLISHED_CELLS)
    assert not pilot._lost(pilot_rows).any()
    record = json.loads((study.STUDY.artifacts / "pilot.json").read_text())
    assert record["rows"] == expected and record["lost_rows"] == 0
    assert record["draws"] == pilot.PILOT_DRAWS and record["pilot_seed"] == pilot.PILOT_SEED


@pytest.fixture(scope="module")
def readings(pilot_rows: pd.DataFrame) -> pd.DataFrame:
    return pilot.pilot_readings(pilot_rows).set_index("cell")


def test_the_predicted_pass_probabilities_are_the_pilot_s(readings: pd.DataFrame) -> None:
    gated = readings.loc[readings["role"] != "diagnostic", "p_passed"]
    assert set(gated.index) == set(study.PREDICTED_PASS)
    for cell, value in study.PREDICTED_PASS.items():
        assert gated[cell] == pytest.approx(value, abs=5e-4), cell


def test_the_predicted_reds(pilot_rows: pd.DataFrame) -> None:
    """Both rules name the same rows: predictive below 0.50, and plug-in below 0.90."""
    positives = [c for c, r in _roles().items() if r == "positive"]
    predictive = {c for c in positives if study.PREDICTED_PASS[c] < study.PREDICTED_RED_BELOW}
    plug_in = {c for c in positives if pilot.plug_in_pass(pilot_rows, c) < 0.90}
    assert predictive == plug_in == set(study.PREDICTED_REDS)
    # Every control is predicted to do its job.
    controls = [c for c, r in _roles().items() if r == "control"]
    assert min(study.PREDICTED_PASS[c] for c in controls) >= 0.98


def _committed_summary() -> pd.DataFrame:
    return pd.read_csv(study.STUDY.artifact("properties.csv"))


def test_the_registered_run_has_no_red_row() -> None:
    """The predicted reds passed, so the study holds no row in the red-cell ledger."""
    from tests.studies.evidence import red_cells

    assert red_cells.red_rows(study.STUDY, red_cells.frames(study.STUDY)) == []
    summary = _committed_summary()
    assert summary["passed"].all()
    assert int(summary["failed_replicates"].sum()) == 0
    for cell in study.PREDICTED_REDS:
        assert bool(summary.set_index("cell").loc[cell, "passed"]), cell


def test_every_gated_configuration_removes_its_caveat() -> None:
    """The declared consequence of a run where every clause passes: no shortfall anywhere.

    Each stacked count of both laws and the cross-fitted ``LTMLE`` cell remove the caveat, so
    no configuration takes an ``adaptive_nuisance`` shortfall and the in-sample sentence alone
    stays (:data:`cleverly._support.ADAPTIVE_IN_SAMPLE_REASON`).
    """
    summary = _committed_summary()
    outcomes = properties.caveat_outcomes(summary)
    expected = {
        study.cell_name("tmle_stacked", law, clusters, "t_reference")
        for law in study.POINT_LAWS
        for clusters in study.COUNTS[law]
    } | {study.cell_name("ltmle_crossfit", study.LTMLE_LAW, 20, "t_reference")}
    assert set(outcomes) == expected
    for cell, outcome in outcomes.items():
        assert outcome["caveat_removed"], cell
        clauses = ("g1", "g2", "variance_ratio", "shown_coverage")
        assert all(outcome[name] for name in clauses), cell
        assert outcome["noninferiority"] is (True if cell.split("__")[1] == "equal10" else None)


def _roles() -> dict[str, str]:
    return {cell.cell: cell.role for cell in properties.declared_cells()}


def test_the_stacked_control_has_power_at_every_count() -> None:
    """Plan 5.4: power at least 0.90 at every count, so no count keeps the caveat for want of it."""
    for law in study.POINT_LAWS:
        for clusters in study.COUNTS[law]:
            cell = study.cell_name("stacked_control", law, clusters, "stacked_control")
            assert study.PREDICTED_PASS[cell] >= 0.90, cell


def test_no_stacked_departure_from_the_parametric_baseline(pilot_rows: pd.DataFrame) -> None:
    """The diagnosis the declaration needed: on every stacked row the variance ratio's 99%
    interval holds 1, and the exact 99% interval of the paired all-draw coverage difference to
    the stacked GLM arm on the same draws holds 0. Mutation: SEs scaled by 0.8 move the ratio's
    interval off 1."""
    groups = dict(tuple(pilot_rows.groupby("cell")))
    for law in study.POINT_LAWS:
        for clusters in study.COUNTS[law]:
            cell = study.cell_name("tmle_stacked", law, clusters, "t_reference")
            group = groups[cell]
            ratio, se = pilot.log_variance_ratio(group["estimate"], group["std_error"])
            assert abs(ratio) < pilot.Z_99 * se, cell
            shrunk, _ = pilot.log_variance_ratio(group["estimate"], 0.8 * group["std_error"])
            assert shrunk < -pilot.Z_99 * se, cell
            counts = properties.paired_counts(group, groups[properties.glm_cell(cell)])
            low = properties.mcnemar_lower(counts["gained"], counts["lost"], counts["trials"])
            high = -properties.mcnemar_lower(counts["lost"], counts["gained"], counts["trials"])
            assert low < 0.0 < high, cell


def test_the_ten_cluster_deficit_is_shared_with_the_glm_arm(pilot_rows: pd.DataFrame) -> None:
    """At 10 clusters the default library and the stacked GLM arm read the same coverage."""
    groups = dict(tuple(pilot_rows.groupby("cell")))
    for law in study.POINT_LAWS:
        cell = study.cell_name("tmle_stacked", law, 10, "t_reference")
        mine = float((groups[cell]["covered"] > 0).mean())
        theirs = float((groups[properties.glm_cell(cell)]["covered"] > 0).mean())
        assert mine < 0.93 and theirs < 0.93
        assert abs(mine - theirs) <= 0.01


def test_no_cell_is_raised_to_4800(pilot_rows: pd.DataFrame) -> None:
    """Plan 5.4 raises a cell whose at-target figure for any clause is below 0.90."""
    for cell, role in _roles().items():
        if role != "positive":
            continue
        figures = pilot.at_target_pass(pilot_rows, cell)
        values = [v for v in figures.values() if np.isfinite(v)]
        assert min(values) >= 0.999, (cell, figures)
        assert np.isfinite(figures["noninferiority"]) == properties.noninferiority_gated(cell)


def test_the_primary_count_is_the_arm_pooled_rule(pilot_rows: pd.DataFrame) -> None:
    assert pilot.primary_replicates(pilot_rows) == study.PRIMARY_REPLICATES == 1_200
    at = pilot.primary_pass(pilot_rows, study.PRIMARY_REPLICATES)
    assert at["family"] == pytest.approx(0.956, abs=1e-3)
    assert pilot.primary_pass(pilot_rows, 1_000)["family"] < pilot.FAMILY_TARGET


def test_the_projected_budget(pilot_rows: pd.DataFrame) -> None:
    """About 279 CPU-h at the pilot's CPU times; the declared per-fit figures within 1.5 times."""
    hours = pilot.projected_cpu_hours(pilot_rows)
    assert hours["total"] == pytest.approx(278.2, abs=1.0)
    assert hours["tmle_stacked"] == pytest.approx(185.8, abs=1.0)
    seconds = pilot.cpu_seconds(pilot_rows)
    planned = {10: 17.8, 20: 20.6, 40: 23.3}
    for law in study.POINT_LAWS:
        for clusters, figure in planned.items():
            assert seconds[("tmle_stacked", law, clusters)] <= 1.5 * figure
    assert seconds[("ltmle_crossfit", "equal40", 20)] <= 1.5 * 26


def test_the_pass_probabilities_see_a_shortfall() -> None:
    """Mutation controls of the predictive probabilities."""
    assert pilot.lower_end_pass(58, 60, 4_000, 0.90) > 0.9
    assert pilot.lower_end_pass(52, 60, 4_000, 0.90) < 0.3
    assert pilot.upper_end_fails(30, 60, 4_000, 0.90) > 0.99
    assert pilot.upper_end_fails(57, 60, 4_000, 0.90) < 0.1
    assert pilot.normal_pass(0.0, 0.05, 0.01, -0.13) > 0.9
    assert pilot.normal_pass(-0.2, 0.05, 0.01, -0.13) < 0.1
