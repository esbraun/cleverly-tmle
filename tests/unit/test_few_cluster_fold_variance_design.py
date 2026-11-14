"""The design of ``few-cluster-fold-variance``: its arms, statistic, rules, pilot and sizing.

The arms are checked against the package's own variance functions on one draw, so a wrong arm
cannot pass by agreeing with itself. The pilot (:mod:`tests.studies.few_cluster_fold_variance_pilot`)
is read from its committed rows, and every declared sizing figure and pilot decision is
recomputed from it. The declared rules (:mod:`tests.studies.few_cluster_fold_variance_properties`)
are checked on constructed rows, each against a mutation that must turn it red.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.inference import influence_variance
from cleverly.inference.cluster import cross_validated_variance
from tests.studies import few_cluster_fold_variance as study
from tests.studies import few_cluster_fold_variance_pilot as pilot
from tests.studies import few_cluster_fold_variance_properties as properties
from tests.studies.evidence import property_failures
from tests.studies.evidence.registry import registered

PILOT = study.STUDY.artifacts / "pilot.csv.gz"


# ------------------------------------------------------------------------------- the arms


@pytest.fixture(scope="module")
def drawn() -> tuple[study.Cell, pd.DataFrame, float]:
    cell = study.CELL_BY_NAME["cluster_covariate__v2_j8"]
    frame = study.draw(cell.law, cell.clusters, np.random.default_rng(5))
    return cell, frame, study.law_truth(cell.law)[study.TARGETS[cell.law]]


def test_the_positive_arm_is_the_centred_fold_cluster_variance(drawn) -> None:
    cell, frame, truth = drawn
    result = study.fit(frame, cell.folds, (study.TARGETS[cell.law],))
    estimate = result[study.TARGETS[cell.law]]
    folds = [test for _, test in result.nuisance.folds]
    cluster = frame["cluster"].to_numpy()
    expected = cross_validated_variance(estimate.influence_curve, folds, cluster)
    arms = study.draw_arms(frame, cell, truth)
    assert arms["t_reference"]["std_error"] ** 2 == pytest.approx(expected, rel=1e-12)
    # The study's interval is the package's own: t with J - V degrees of freedom.
    assert estimate.reference_df == cell.df == 6
    low, high = estimate.plugin_interval
    assert arms["t_reference"]["low"] == pytest.approx(low, abs=1e-12)
    assert arms["t_reference"]["high"] == pytest.approx(high, abs=1e-12)
    # Mutation control: rows as units, the IID arm, is a different number on this law.
    assert arms["iid_t_control"]["std_error"] < 0.9 * arms["t_reference"]["std_error"]


def test_the_pooled_arm_recentres_the_fold_curves_at_the_pooled_point(drawn) -> None:
    """Witness: the stacked variance equals the fold curves shifted by their fold points."""
    cell, frame, truth = drawn
    target = study.TARGETS[cell.law]
    result = study.fit(frame, cell.folds, (target,))
    estimate = result[target]
    folds = [test for _, test in result.nuisance.folds]
    shifted = np.asarray(estimate.influence_curve, dtype=float).copy()
    for value, test in zip(result.cv_targeting.fold_estimates[target], folds, strict=True):
        shifted[test] += value
    cluster = frame["cluster"].to_numpy()
    arms = study.draw_arms(frame, cell, truth)
    assert arms["pooled_centered"]["std_error"] ** 2 == pytest.approx(
        influence_variance(shifted, cluster), rel=1e-10
    )
    # Mutation control: the unshifted curve gives the fold-centred spread, not the pooled one.
    assert influence_variance(estimate.influence_curve, cluster) != pytest.approx(
        influence_variance(shifted, cluster), rel=1e-6
    )


def test_the_deflated_and_reference_arms_keep_the_point(drawn) -> None:
    cell, frame, truth = drawn
    arms = study.draw_arms(frame, cell, truth)
    base = arms["t_reference"]
    assert arms["deflated_control"]["std_error"] == pytest.approx(
        base["std_error"] * 0.80, rel=1e-12
    )
    known = arms["known_propensity"]
    assert arms["known_deflated_control"]["std_error"] == pytest.approx(
        known["std_error"] * 0.80, rel=1e-12
    )
    for arm in study.ARMS[:6]:
        assert arms[arm]["psi"] == base["psi"]
    # The known-propensity arms are a second fit: another point on the same draw.
    assert known["psi"] != base["psi"]
    assert arms["known_deflated_control"]["psi"] == known["psi"]


def test_the_jensen_factor_matches_the_plan() -> None:
    factors = [study.jensen_factor(k) for k in (2, 5, 15, 25)]
    assert factors == pytest.approx([0.886, 0.951, 0.984, 0.990], abs=1e-3)


def test_the_truths_are_the_laws_of_the_fix4_studies() -> None:
    from tests.studies.clustered_few_cluster_properties import SizeLaw

    assert study.law_truth("equal10")["ate"] == SizeLaw("equal10/J10").truth()["ate"]
    assert study.TARGETS == {"equal10": "ate", "cluster_covariate": "ey1"}


# -------------------------------------------------------------------------- the statistic


def test_the_variance_ratio_is_the_mean_squared_se_over_the_spread() -> None:
    rng = np.random.default_rng(0)
    estimates, std_errors = rng.normal(size=500), rng.uniform(0.5, 1.5, size=500)
    assert properties.variance_ratio(estimates, std_errors) == pytest.approx(
        np.mean(std_errors**2) / np.var(estimates, ddof=1), rel=1e-14
    )


def test_the_bootstrap_resamples_replicates_jointly() -> None:
    """An SE tied to its own estimate moves the interval; resampled apart it would not."""
    rng = np.random.default_rng(1)
    estimates = rng.normal(size=400)
    tied = np.abs(estimates) + 0.5
    loose = rng.permutation(tied)
    kwargs = {"replicates": 2_000, "confidence_level": 0.99, "seed": 3}
    a = properties.variance_ratio_interval(estimates, tied, **kwargs)
    b = properties.variance_ratio_interval(estimates, loose, **kwargs)
    assert properties.variance_ratio(estimates, tied) == pytest.approx(
        properties.variance_ratio(estimates, loose)
    )
    assert (a.high - a.low) != pytest.approx(b.high - b.low, rel=1e-3)


# ------------------------------------------------------------------------------ the record


def test_the_record_publishes_every_declared_cell() -> None:
    declared = {cell.cell for cell in properties.declared_cells()}
    assert set(study.STUDY.property_cells[study.FAMILY]) == declared
    assert len(declared) == len(study.CELLS) * len(study.ARMS) == 96
    roles = {cell.cell: cell.role for cell in properties.declared_cells()}
    assert {n for n, r in roles.items() if r == "positive"} == set(properties.POSITIVE_CELLS)
    assert {n for n, r in roles.items() if r == "control"} == {
        *properties.IID_CELLS,
        *properties.DEFLATED_CELLS,
        *properties.POOLED_CELLS,
    }
    for cell in properties.declared_cells():
        assert cell.replicates == study.PROPERTY_REPLICATES[cell.dgp.name]


def test_every_design_has_equal_folds_of_two_four_or_six_clusters() -> None:
    assert {(c.folds, c.per_fold) for c in study.CELLS} == {
        (v, m) for v in (2, 5) for m in (2, 4, 6)
    }
    assert all(c.clusters % c.folds == 0 for c in study.CELLS)


def test_the_cells_are_read_by_training_clusters() -> None:
    """J (V - 1) / V: 2, 4, 6 at V = 2 and 8, 16, 24 at V = 5, one design per count."""
    assert study.TRAINING_LADDER == (2, 4, 6, 8, 16, 24)
    by_count = {c.training: (c.folds, c.clusters) for c in study.CELLS}
    assert by_count == {2: (2, 4), 4: (2, 8), 6: (2, 12), 8: (5, 10), 16: (5, 20), 24: (5, 30)}


def test_the_seeds_are_the_study_s_own() -> None:
    seeds = {
        s
        for record in registered()
        if record.slug != study.STUDY.slug
        for s in (record.seed, record.resampling_seed)
    }
    assert not {study.SEED, study.RESAMPLING_SEED} & seeds
    pilot_seeds = {pilot.pilot_seed(c.name, r) for c in study.CELLS for r in range(3)}
    property_seeds = {properties._seed(c.name, r) for c in study.CELLS for r in range(3)}
    assert not pilot_seeds & property_seeds


def test_registered_from_the_declared_run() -> None:
    from tests.unit.test_method_evidence import UNCOMMITTED_MODULES

    assert study.STUDY.slug in {record.slug for record in registered()}
    assert study.STUDY.artifact("property-replicates.parquet").exists()
    assert properties.__name__ not in UNCOMMITTED_MODULES
    assert study.STUDY.publication_policy == "reporting"
    assert properties.FAILURE_BOUND == 0.05
    assert study.STUDY.failure_bound == {"cleverly": 0.01}


# ------------------------------------------------------------------------------- the pilot


@pytest.fixture(scope="module")
def rows() -> pd.DataFrame:
    return pilot.read(PILOT)


@pytest.fixture(scope="module")
def readings(rows: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    return {arm: pilot.readings_of(rows, arm) for arm in study.POSITIVE_ARMS}


def test_the_pilot_holds_every_cell_and_the_declared_extensions(rows: pd.DataFrame) -> None:
    """The extension to 1,600 draws was decided under rev 3.1's rule, at commit 7c8d319a."""
    counts = rows.groupby("cell").size().to_dict()
    assert set(counts) == {cell.name for cell in study.CELLS}
    for name, count in counts.items():
        expected = (
            pilot.EXTENDED_REPLICATES if name in study.PILOT_EXTENDED else pilot.PILOT_REPLICATES
        )
        assert count == expected, name
        assert sorted(rows.loc[rows["cell"] == name, "replicate"]) == list(range(expected))


def test_no_pilot_draw_is_lost_under_the_constant_target_rule(rows: pd.DataFrame) -> None:
    """The 43 draws a missing outcome class lost were refitted; no complement lacked an arm."""
    assert (rows["lost"] == "").all()
    assert (rows.loc[rows["cell"] == "equal10__v2_j4", "t_reference_withheld"] != "").sum() == 86


def test_the_known_deflated_arm_is_derived_exactly(rows: pd.DataFrame) -> None:
    kept = rows["known_propensity_withheld"] == ""
    assert np.allclose(
        rows["known_deflated_control_std_error"], rows["known_propensity_std_error"] * 0.80
    )
    # Mutation control: deflating the SE can only lose coverage.
    assert (
        rows.loc[kept, "known_deflated_control_covered"]
        <= rows.loc[kept, "known_propensity_covered"]
    ).all()
    assert rows["known_deflated_control_covered"].sum() < rows["known_propensity_covered"].sum()


def test_the_declared_replicates_are_8000_everywhere() -> None:
    assert set(study.PROPERTY_REPLICATES.values()) == {pilot.REPLICATES} == {8_000}


@pytest.mark.parametrize("arm", study.POSITIVE_ARMS)
def test_the_predicted_reds_and_floors_are_the_declared_ones(readings, arm: str) -> None:
    """Cells whose plug-in pass probability at 8,000 draws is below 0.90, and the floor above."""
    assert pilot.predicted_red(readings[arm]) == study.PREDICTED_RED[arm]
    assert pilot.predicted_floor(readings[arm]) == study.PREDICTED_FLOOR[arm]


@pytest.mark.parametrize(
    ("arm", "at_4000", "at_8000"),
    [("t_reference", 0.970, 1.000), ("known_propensity", 0.847, 0.997)],
)
def test_the_family_products_are_the_plan_s(
    rows, readings, arm: str, at_4000: float, at_8000: float
) -> None:
    """Plan 4.6 and 5.3: the product above each predicted floor, controls included."""
    floor = study.PREDICTED_FLOOR[arm]
    assert pilot.floor_product(readings[arm], floor, 4_000) == pytest.approx(at_4000, abs=2e-3)
    assert pilot.floor_product(readings[arm], floor) == pytest.approx(at_8000, abs=2e-3)
    family = pilot.family_with_controls(rows, arm, floor)
    assert family == pytest.approx(at_8000, abs=2e-3)
    assert family >= 0.95


@pytest.mark.parametrize("arm", study.POSITIVE_ARMS)
def test_the_predictive_family_figures_are_reported(readings, arm: str) -> None:
    """Plan section 5: the Jeffreys-predictive figure beside the plug-in one."""
    floor = study.PREDICTED_FLOOR[arm]
    predictive = pilot.predictive_product(readings[arm], floor)
    assert predictive == pytest.approx(study.PREDICTIVE_FAMILY[arm], abs=2e-3)
    # The predictive figure carries the pilot's own uncertainty, so it sits below the plug-in.
    assert predictive < pilot.floor_product(readings[arm], floor)


@pytest.mark.parametrize("arm", study.POSITIVE_ARMS)
def test_the_deflated_controls_fail_the_rule_in_every_cell(rows, arm: str) -> None:
    """At 0.80 squared each control fails the rule with power at least 0.999 in every cell."""
    control = study.DEFLATED_ARMS[arm]
    for name, reading in pilot.readings_of(rows, control).items():
        assert pilot.control_power(reading) >= 0.999, name


def test_the_deflation_was_sized_from_the_pilot(rows, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation control: at 0.85 squared the estimated ``equal10__v2_j12`` control is a coin flip.

    The positive there is conservative (variance ratio 1.31), so a 15% narrower interval still
    covers 0.927 and meets the rule; the deflation has to be larger, not R.
    """
    monkeypatch.setattr(study, "DEFLATION", 0.85**2)
    weak = pilot.derive_deflated(rows)
    reading = pilot.shapes(weak, "equal10__v2_j12", "deflated_control")
    assert 0.4 < pilot.control_power(reading) < 0.7


def test_the_red_cell_owner_is_listed_and_holds_every_predicted_red() -> None:
    from tests.studies.evidence.red_cells import owner_ids

    assert study.RED_OWNER in owner_ids()
    predicted = {f"{cell}__{arm}" for arm, cells in study.PREDICTED_RED.items() for cell in cells}
    assert set(study.PREDICTED_RED_OWNERS) == predicted
    assert set(study.PREDICTED_RED_OWNERS.values()) == {study.RED_OWNER}
    # Every predicted red sits below its configuration's predicted floor.
    for arm, cells in study.PREDICTED_RED.items():
        for cell in cells:
            assert study.CELL_BY_NAME[cell].training < study.PREDICTED_FLOOR[arm]


def test_the_known_configuration_is_the_declared_mechanism(drawn) -> None:
    """Witness: ``treatment_probabilities=`` and the pilot's oracle learner give the same fit."""
    cell, frame, _ = drawn
    target = study.TARGETS[cell.law]
    declared = study.fit(frame, cell.folds, (target,), known=cell.law)
    oracle = study.fit(frame, cell.folds, (target,), known=cell.law, oracle=True)
    assert declared[target].psi == pytest.approx(oracle[target].psi, abs=1e-12)
    assert declared[target].plugin_std_error == pytest.approx(
        oracle[target].plugin_std_error, abs=1e-12
    )
    assert declared[target].inference == oracle[target].inference
    assert declared.nuisance.treatment_mechanism == "known"
    assert oracle.nuisance.treatment_mechanism == "estimated"
    # Mutation control: the fitted propensity gives another point on the same draw.
    estimated = study.fit(frame, cell.folds, (target,))
    assert estimated[target].psi != pytest.approx(declared[target].psi, abs=1e-6)


def test_the_iid_control_has_power_in_every_cell(rows: pd.DataFrame, readings) -> None:
    """The median form: power at least 0.99 everywhere, where the mean form had none at (2, 4)."""
    for cell in study.CELLS:
        assert pilot.iid_median_power(rows, cell.name, pilot.REPLICATES) >= 0.99, cell.name
    # The mean form at 2 training clusters: a heavy lower tail of the cluster SE drives it up.
    assert readings["t_reference"]["cluster_covariate__v2_j4"]["iid_mean"] > 0.6
    assert readings["t_reference"]["cluster_covariate__v2_j4"]["iid_median"] < 0.5


def test_the_pooled_centred_gate_decision(readings) -> None:
    """No cell gates ``pooled_centered``: re-centring adds no excess to the fold variance.

    The pilot's ratio of the pooled-centred to the fold variance ratio stays below 1.25 in every
    cell (plan trigger), including the cells with 2 clusters per fold.
    """
    assert frozenset() == study.POOLED_CENTERED_GATED
    for name, reading in readings["t_reference"].items():
        assert reading["pooled_ratio"] / reading["ratio"] < study.POOLED_CENTERED_PILOT_TRIGGER, (
            name
        )


def test_the_closed_form_sizing_matches_the_plan_s_simulation() -> None:
    """Witness: the one-sided closed form against ``variance_ratio_sim`` (plan rev 2, k = 2)."""
    for kurt_se, kurt_psi, replicates, ratio in (
        (2.0, 1.0, 4_000, 0.92),
        (2.0, 1.0, 8_000, 0.90),
        (0.5, 0.0, 4_000, 0.89),
    ):
        relvar = 2.0 / 2.0 + kurt_se
        closed = pilot.variance_lower_pass(relvar, kurt_psi, replicates, ratio)
        simulated = pilot.variance_ratio_sim(
            2, kurt_se, replicates, kurt_psi=kurt_psi, true_ratio=ratio, sims=400
        )
        assert closed == pytest.approx(simulated, abs=0.07)
    # Mutation control: an interval that ignored the spread of the squared SE passes more often.
    assert pilot.variance_lower_pass(0.0, 0.0, 4_000, 0.90) > pilot.variance_lower_pass(
        3.0, 1.0, 4_000, 0.90
    )


# ------------------------------------------------------------------------------ the rules

#: One cell keeps the constructed rows small; every rule reads each cell on its own.
RULE_CELL = "cluster_covariate__v2_j8"
POSITIVE = study.cell_name(RULE_CELL, "t_reference")
DEFLATED = study.cell_name(RULE_CELL, "deflated_control")
IID = study.cell_name(RULE_CELL, "iid_t_control")
KNOWN = study.cell_name(RULE_CELL, "known_propensity")


KNOWN_DEFLATED = study.cell_name(RULE_CELL, "known_deflated_control")


def _rows(seed: int = 0, replicates: int = 1_200) -> pd.DataFrame:
    """Rows of one cell where every gated rule holds by a wide margin.

    The estimates are standardized, so the variance ratio of an arm is its mean squared SE.
    """
    rng = np.random.default_rng(seed)
    cell = study.CELL_BY_NAME[RULE_CELL]
    errors = rng.normal(size=replicates)
    errors = (errors - errors.mean()) / errors.std(ddof=1)
    covered = (np.arange(replicates) % 20 != 0).astype(int)
    out = []
    for arm in study.ARMS:
        se = {"iid_t_control": 0.6, "deflated_control": 0.80, "known_deflated_control": 0.80}
        deflated = arm in study.DEFLATED_ARMS.values()
        out.append(
            pd.DataFrame(
                {
                    "property": study.FAMILY,
                    "cell": study.cell_name(RULE_CELL, arm),
                    "role": study.arm_role(RULE_CELL, arm),
                    "replicate": np.arange(replicates),
                    "n": cell.n,
                    "requested_replicates": replicates,
                    "failed_replicates": 0,
                    "truth": 0.0,
                    "estimate": errors,
                    "std_error": se.get(arm, 1.0) * rng.uniform(0.98, 1.02, size=replicates),
                    # A deflated interval covers less: 0.90 here.
                    "covered": (np.arange(replicates) % 10 != 0).astype(int)
                    if deflated
                    else covered,
                    "rejected": 0,
                    "withheld": 0,
                    "reference_df": cell.df,
                }
            )
        )
    return pd.concat(out, ignore_index=True)


def _verdicts(rows: pd.DataFrame) -> pd.DataFrame:
    summary = properties.summarize_properties(rows)
    ctx = property_failures.Context(
        record=study.STUDY, rows=rows, summary=summary, groups=properties.PROPERTY_GROUPS
    )
    rule = property_failures.rule_verdicts(properties.PROPERTY_RULES[0], summary, ctx)
    merged = summary.merge(rule, on=["property", "cell"], suffixes=("", "_rule"))
    # The declared rule restates the module's verdict on every verdict row.
    assert (merged["passed"] == merged["passed_rule"]).all()
    assert (merged["property_passed"] == merged["property_passed_rule"]).all()
    return summary.set_index("cell")


@pytest.fixture(scope="module")
def passing() -> pd.DataFrame:
    return _rows()


def _scaled(rows: pd.DataFrame, cell: str, factor: float) -> pd.DataFrame:
    out = rows.copy()
    out.loc[out["cell"] == cell, "std_error"] *= factor
    return out


def test_the_rules_pass_rows_that_meet_them(passing: pd.DataFrame) -> None:
    assert _verdicts(passing)["passed"].all()


def test_a_variance_above_the_band_is_reported_not_gated(passing: pd.DataFrame) -> None:
    """Rev 3.2: the upper side measures outlier SEs; the coverage band gates typical draws."""
    verdicts = _verdicts(_scaled(passing, POSITIVE, 1.2))
    assert verdicts.loc[POSITIVE, "variance_ratio_ci_lower"] > properties.VARIANCE_BAND[1]
    assert verdicts.loc[POSITIVE, "passed"]
    # Mutation control: the same variance with the over-coverage it brings on typical draws.
    rows = _scaled(passing, POSITIVE, 1.2)
    rows.loc[rows["cell"] == POSITIVE, "covered"] = 1
    assert not _verdicts(rows).loc[POSITIVE, "passed"]


@pytest.mark.parametrize("cell", [POSITIVE, KNOWN])
def test_a_variance_below_the_band_turns_a_positive_red_with_an_se_ratio_inside(
    passing: pd.DataFrame, cell: str
) -> None:
    """Mutation control: a variance ratio of 0.76 reads an SE ratio of 0.87, inside 0.8 to 1.2."""
    verdicts = _verdicts(_scaled(passing, cell, 0.87))
    assert 0.8 < verdicts.loc[cell, "shown_se_ratio"] < 1.2
    assert verdicts.loc[cell, "variance_ratio_ci_upper"] < properties.VARIANCE_BAND[0]
    assert not verdicts.loc[cell, "passed"]


def test_g1_counts_withheld_draws_as_misses(passing: pd.DataFrame) -> None:
    """Seven per cent withheld leaves the shown band green and puts G1's lower end below 0.90."""
    rows = passing.copy()
    mask = (rows["cell"] == POSITIVE) & (rows["replicate"] % 100 < 7)
    rows.loc[mask, ["covered", "withheld"]] = [0, 1]
    verdicts = _verdicts(rows)
    assert properties.COVERAGE_BAND[0] <= verdicts.loc[POSITIVE, "shown_ci_lower"]
    assert verdicts.loc[POSITIVE, "all_draw_ci_lower"] < properties.G1_FLOOR
    assert not verdicts.loc[POSITIVE, "passed"]
    # Mutation control: two per cent withheld clears G1.
    rows = passing.copy()
    mask = (rows["cell"] == POSITIVE) & (rows["replicate"] % 100 < 2)
    rows.loc[mask, ["covered", "withheld"]] = [0, 1]
    assert _verdicts(rows).loc[POSITIVE, "passed"]


@pytest.mark.parametrize(("control", "positive"), [(DEFLATED, POSITIVE), (KNOWN_DEFLATED, KNOWN)])
def test_a_deflated_control_fails_when_it_passes_the_positive_rule(
    passing: pd.DataFrame, control: str, positive: str
) -> None:
    """An undeflated control, at the positive's own SE and coverage, passes the rule: red."""
    verdicts = _verdicts(passing)
    assert verdicts.loc[control, "passed"] and not verdicts.loc[control, "positive_rule_passed"]
    rows = _scaled(passing, control, 1 / 0.80)
    rows.loc[rows["cell"] == control, "covered"] = rows.loc[
        rows["cell"] == positive, "covered"
    ].to_numpy()
    verdicts = _verdicts(rows)
    assert verdicts.loc[control, "positive_rule_passed"]
    assert not verdicts.loc[control, "passed"]
    assert verdicts.loc[positive, "passed"]


def test_an_iid_se_like_the_cluster_se_fails_its_control(passing: pd.DataFrame) -> None:
    """An inert aggregation: the IID SE equals the cluster SE, so every paired ratio is 1."""
    rows = passing.copy()
    rows.loc[rows["cell"] == IID, "std_error"] = rows.loc[
        rows["cell"] == POSITIVE, "std_error"
    ].to_numpy()
    verdicts = _verdicts(rows)
    assert verdicts.loc[IID, "paired_median_ratio"] == 1.0
    assert not verdicts.loc[IID, "passed"]
    # Mutation control: a ratio of 0.9 on every draw is under the ceiling.
    rows.loc[rows["cell"] == IID, "std_error"] *= 0.9
    assert _verdicts(rows).loc[IID, "passed"]


def test_the_median_interval_is_the_order_statistic_interval() -> None:
    values = np.arange(1.0, 101.0)
    interval = properties.median_interval(values, confidence_level=0.99)
    # Bin(100, 1/2): the 0.005 quantile is 37, so the interval is [x_(37), x_(64)].
    assert (interval.low, interval.high) == (37.0, 64.0)
    # The accounting's fills: values above every survivor raise the upper end.
    assert properties.median_interval(values, confidence_level=0.99, high=4).high > interval.high


def test_withheld_draws_are_left_out_of_the_shown_statistics(passing: pd.DataFrame) -> None:
    """A withheld draw with a wild SE does not move the ratio; read as shown it would."""
    rows = passing.copy()
    mask = (rows["cell"] == POSITIVE) & (rows["replicate"] < 12)
    rows.loc[mask, "std_error"] = 30.0
    rows.loc[mask, "covered"] = 0
    rows.loc[mask, "withheld"] = 1
    verdicts = _verdicts(rows)
    assert verdicts.loc[POSITIVE, "passed"]
    assert verdicts.loc[POSITIVE, "withheld_draws"] == 12
    # Mutation control: the same rows read as shown carry a ratio far above the band.
    positive = rows.loc[rows["cell"] == POSITIVE]
    assert properties.variance_ratio(positive["estimate"], positive["std_error"]) > 5.0


def test_a_lost_draw_is_read_at_every_split() -> None:
    row = pd.Series({"shown_draws": 1_000, "shown_coverage": 0.95, "failed_replicates": 3})
    ctx = property_failures.Context(record=study.STUDY, rows=pd.DataFrame(), summary=pd.DataFrame())
    candidates = properties._coverage_candidates(ctx, row)
    assert len(candidates) == 10
    assert candidates[0][0] == "0 shown, 0 covered"


def _summary(**columns: Any) -> pd.DataFrame:
    names = [n for c in study.CELLS for n in (study.cell_name(c.name, a) for a in study.ARMS)]
    out = pd.DataFrame(
        {"property": study.FAMILY, "cell": names, "passed": True, "variance_lower_passed": 1.0}
    )
    for key, value in columns.items():
        out[key] = value
    return out


def _fail(summary: pd.DataFrame, cell: str, arm: str, *, variance: bool = True) -> None:
    mask = summary["cell"] == study.cell_name(cell, arm)
    summary.loc[mask, "passed"] = False
    if not variance:
        summary.loc[mask, "variance_lower_passed"] = 0.0


def test_the_training_floor_is_the_smallest_count_from_which_every_cell_passes() -> None:
    summary = _summary()
    assert properties.training_floor(summary, "t_reference") == 2
    _fail(summary, "cluster_covariate__v2_j4", "t_reference")
    assert properties.training_floor(summary, "t_reference") == 4
    _fail(summary, "equal10__v2_j12", "t_reference")
    assert properties.training_floor(summary, "t_reference") == 8
    # The other configuration has its own floor.
    assert properties.training_floor(summary, "known_propensity") == 2
    _fail(summary, "equal10__v5_j30", "known_propensity")
    assert properties.training_floor(summary, "known_propensity") is None


def test_the_route_of_a_configuration_with_no_red_is_calibrated() -> None:
    route = properties.declared_route(_summary(), "t_reference")
    assert (route.outcome, route.floor, route.minimum) == (properties.CALIBRATED, 2, None)


def test_the_route_of_the_predicted_reds_is_the_owned_floor() -> None:
    summary = _summary()
    for arm, cells in study.PREDICTED_RED.items():
        for cell in cells:
            _fail(summary, cell, arm)
    for arm in study.POSITIVE_ARMS:
        route = properties.declared_route(summary, arm)
        assert route.outcome == properties.TRAINING_FLOOR
        assert route.floor == route.minimum == study.PREDICTED_FLOOR[arm]
        held = (study.RED_OWNER, study.TRAINING_FLOOR_READING)
        assert route.owners == {f"{c}__{arm}": held for c in study.PREDICTED_RED[arm]}
        assert route.non_monotone == ()


def test_a_red_variance_lower_end_below_the_floor_is_a_defect_suspect() -> None:
    """Mutation of the owned case: the same red, but an underestimated variance."""
    summary = _summary()
    _fail(summary, "equal10__v2_j4", "t_reference", variance=False)
    route = properties.declared_route(summary, "t_reference")
    assert route.outcome == properties.DEFECT_SUSPECT
    assert route.owners == {"equal10__v2_j4__t_reference": None}


@pytest.mark.parametrize(
    ("cell", "floor"), [("equal10__v5_j10", 16), ("cluster_covariate__v5_j20", 24)]
)
def test_a_chance_red_above_a_passing_count_is_held_as_finite_sample(cell: str, floor: int) -> None:
    """Plan 5.8 (rev 3.5): a red at 8 or 16 above passing counts is the non-monotone signature.

    The floor rises past it and the owner holds it under the finite-sample reading, not the
    training-floor limit, which is monotone in the count.
    """
    summary = _summary()
    _fail(summary, "equal10__v2_j4", "t_reference")
    _fail(summary, cell, "t_reference")
    route = properties.declared_route(summary, "t_reference")
    name = f"{cell}__t_reference"
    assert route.outcome == properties.TRAINING_FLOOR
    assert route.minimum == floor
    assert route.non_monotone == (name,)
    assert route.owners == {
        "equal10__v2_j4__t_reference": (study.RED_OWNER, study.TRAINING_FLOOR_READING),
        name: (study.RED_OWNER, study.FINITE_SAMPLE_READING),
    }


def test_a_red_at_the_predicted_floor_reads_finite_sample_without_the_signature() -> None:
    """A red at the predicted floor with every lower count red is still not the pilot's limit."""
    summary = _summary()
    for cell in ("equal10__v2_j4", "equal10__v2_j8"):
        _fail(summary, cell, "t_reference")
    route = properties.declared_route(summary, "t_reference")
    assert route.non_monotone == ()
    assert route.owners["equal10__v2_j8__t_reference"] == (
        study.RED_OWNER,
        study.FINITE_SAMPLE_READING,
    )
    # The known configuration's predicted floor is 6, so the same red there is the limit.
    for cell in ("equal10__v2_j4", "equal10__v2_j8"):
        _fail(summary, cell, "known_propensity")
    known = properties.declared_route(summary, "known_propensity").owners
    assert known["equal10__v2_j8__known_propensity"] == (
        study.RED_OWNER,
        study.TRAINING_FLOOR_READING,
    )
    # Below the predicted floor, the non-monotone signature alone makes the reading finite-sample.
    summary = _summary()
    _fail(summary, "equal10__v2_j8", "known_propensity")
    route = properties.declared_route(summary, "known_propensity")
    assert route.non_monotone == ("equal10__v2_j8__known_propensity",)
    assert route.owners == {
        "equal10__v2_j8__known_propensity": (study.RED_OWNER, study.FINITE_SAMPLE_READING)
    }


def test_no_passing_count_is_a_defect_suspect() -> None:
    summary = _summary()
    _fail(summary, "equal10__v5_j30", "t_reference")
    route = properties.declared_route(summary, "t_reference")
    assert route.outcome == properties.DEFECT_SUSPECT and route.floor is None
    assert route.owners == {"equal10__v5_j30__t_reference": None}
    # No diagnosis is recorded before the run.
    assert study.NO_DEFECT_DIAGNOSES == {}


def test_a_diagnosed_red_at_24_ships_no_floor() -> None:
    """Plan 5.8 (rev 3.5): the fifth outcome needs the recorded diagnosis flag."""
    summary = _summary()
    _fail(summary, "equal10__v2_j4", "t_reference")
    _fail(summary, "equal10__v5_j30", "t_reference")
    diagnoses = {"t_reference": "no defect: the record of the diagnosis"}
    route = properties.declared_route(summary, "t_reference", diagnoses)
    assert route.outcome == properties.NO_FLOOR
    assert route.floor is None and route.minimum is None
    assert route.reasons == ("no defect: the record of the diagnosis",)
    assert route.owners == {
        "equal10__v2_j4__t_reference": (study.RED_OWNER, study.TRAINING_FLOOR_READING),
        "equal10__v5_j30__t_reference": (study.RED_OWNER, study.FINITE_SAMPLE_READING),
    }
    assert route.non_monotone == ("equal10__v5_j30__t_reference",)


def test_the_no_floor_outcome_needs_its_own_diagnosis_and_a_passing_variance() -> None:
    """Mutations of the no-floor case: each removes one condition and leaves a defect suspect."""
    summary = _summary()
    _fail(summary, "equal10__v5_j30", "t_reference")
    other = {"known_propensity": "no defect"}
    assert properties.declared_route(summary, "t_reference", other).outcome == (
        properties.DEFECT_SUSPECT
    )
    own = {"t_reference": "no defect"}
    _fail(summary, "equal10__v5_j30", "t_reference", variance=False)
    route = properties.declared_route(summary, "t_reference", own)
    assert route.outcome == properties.DEFECT_SUSPECT
    assert route.owners == {"equal10__v5_j30__t_reference": None}
    # A diagnosis never moves a configuration that has a floor.
    summary = _summary()
    _fail(summary, "equal10__v2_j4", "t_reference")
    assert properties.declared_route(summary, "t_reference", own).outcome == (
        properties.TRAINING_FLOOR
    )


@pytest.mark.parametrize(
    ("cell", "arm"),
    [
        ("equal10__v2_j12", "deflated_control"),
        ("cluster_covariate__v5_j30", "deflated_control"),
        ("equal10__v2_j8", "iid_t_control"),
    ],
)
def test_a_control_that_does_not_do_its_job_withholds_the_licence(cell: str, arm: str) -> None:
    """Plan 5.8: a must-fail control that passes the rule means nothing ships from the study."""
    summary = _summary()
    _fail(summary, cell, arm)
    route = properties.declared_route(summary, "t_reference")
    assert route.outcome == properties.UNLICENSED and route.minimum is None
    # Mutation control: the other configuration's deflated control is not this one's licence.
    if arm == "deflated_control":
        assert properties.declared_route(summary, "known_propensity").outcome == (
            properties.CALIBRATED
        )


def test_the_pilot_file_is_committed() -> None:
    assert Path(PILOT).exists()


# ------------------------------------------------------------------------- the declared run


def test_the_committed_run_takes_the_declared_routes() -> None:
    """The registered verdicts, read through ``declared_route``, and the ledger's owner keys."""
    from tests.studies.evidence.red_cells import CLAIMS

    summary = pd.read_csv(study.STUDY.artifact("properties.csv"))
    held = (study.RED_OWNER, study.TRAINING_FLOOR_READING)
    owned: set[str] = set()
    for arm in study.POSITIVE_ARMS:
        route = properties.declared_route(summary, arm)
        assert (route.outcome, route.floor, route.minimum) == (properties.TRAINING_FLOOR, 4, 4)
        assert route.non_monotone == () and route.reasons == ()
        assert set(route.owners.values()) == {held}
        owned |= set(route.owners)
    claimed = {key.key.split("/", 1)[1] for key in CLAIMS[study.RED_OWNER]}
    assert owned == claimed
    red = set(summary.loc[~summary["passed"].astype(bool), "cell"])
    assert red == owned


def test_the_predicted_reds_that_passed_are_the_known_floor_s_two() -> None:
    """The pilot predicted a known floor of 6; two of its predicted reds passed, so it is 4."""
    summary = pd.read_csv(study.STUDY.artifact("properties.csv")).set_index("cell")
    passed = summary["passed"].astype(bool)
    missed = {
        arm: {c for c in cells if passed[study.cell_name(c, arm)]}
        for arm, cells in study.PREDICTED_RED.items()
    }
    assert missed == {
        "t_reference": set(),
        "known_propensity": {"cluster_covariate__v2_j4", "equal10__v2_j8"},
    }
