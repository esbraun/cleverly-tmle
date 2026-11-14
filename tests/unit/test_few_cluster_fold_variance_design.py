"""The design of ``few-cluster-fold-variance``: its arms, statistic, rules, pilot and sizing.

The arms are checked against the package's own variance functions on one draw, so a wrong arm
cannot pass by agreeing with itself. The pilot (:mod:`tests.studies.few_cluster_fold_variance_pilot`)
is read from its committed rows, and every declared sizing figure and pilot decision is
recomputed from it. The declared rules (:mod:`tests.studies.few_cluster_fold_variance_properties`)
are checked on constructed rows, each against a mutation that must turn it red.
"""

from __future__ import annotations

from pathlib import Path

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
        base["std_error"] * 0.85, rel=1e-12
    )
    for arm in study.ARMS[:-1]:
        assert arms[arm]["psi"] == base["psi"]
    # The known-propensity arm is a second fit: another point on the same draw.
    assert arms["known_propensity"]["psi"] != base["psi"]


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
    assert len(declared) == len(study.CELLS) * len(study.ARMS) == 84
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


def test_the_seeds_are_the_study_s_own() -> None:
    seeds = {s for record in registered() for s in (record.seed, record.resampling_seed)}
    assert not {study.SEED, study.RESAMPLING_SEED} & seeds
    pilot_seeds = {pilot.pilot_seed(c.name, r) for c in study.CELLS for r in range(3)}
    property_seeds = {properties._seed(c.name, r) for c in study.CELLS for r in range(3)}
    assert not pilot_seeds & property_seeds


def test_declared_and_not_yet_registered() -> None:
    from tests.unit.test_method_evidence import UNCOMMITTED_MODULES

    assert study.STUDY.slug not in {record.slug for record in registered()}
    assert not study.STUDY.artifact("property-replicates.parquet").exists()
    assert properties.__name__ in UNCOMMITTED_MODULES
    assert study.STUDY.publication_policy == "reporting"
    assert properties.FAILURE_BOUND == 0.05
    assert study.STUDY.failure_bound == {"cleverly": 0.05}


# ------------------------------------------------------------------------------- the pilot


@pytest.fixture(scope="module")
def rows() -> pd.DataFrame:
    return pilot.read(PILOT)


@pytest.fixture(scope="module")
def readings(rows: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {cell.name: pilot.shapes(rows, cell.name) for cell in study.CELLS}


def test_the_pilot_holds_every_cell_and_the_declared_extensions(rows: pd.DataFrame) -> None:
    counts = rows.groupby("cell").size().to_dict()
    assert set(counts) == {cell.name for cell in study.CELLS}
    for name, count in counts.items():
        expected = (
            pilot.EXTENDED_REPLICATES if name in study.PILOT_EXTENDED else pilot.PILOT_REPLICATES
        )
        assert count == expected, name
        assert sorted(rows.loc[rows["cell"] == name, "replicate"]) == list(range(expected))


def test_the_extended_cells_are_those_below_090_at_400_draws(rows: pd.DataFrame) -> None:
    """The guide's rule: extend every cell whose plug-in pass probability is below 0.90."""
    first = rows.loc[rows["replicate"] < pilot.PILOT_REPLICATES]
    early = {cell.name: pilot.shapes(first, cell.name) for cell in study.CELLS}
    sized = pilot.declared_replicates(early)
    below = {
        name for name, reading in early.items() if pilot.plug_in_pass(reading, sized[name]) < 0.90
    }
    assert below == set(study.PILOT_EXTENDED)


def test_the_declared_replicates_are_the_sizing_rule(readings) -> None:
    assert pilot.shortfall(readings) == study.PREDICTED_SHORTFALL
    sized = pilot.declared_replicates(readings)
    assert sized == study.PROPERTY_REPLICATES
    kept = [name for name in sized if name not in study.PREDICTED_SHORTFALL]
    for name in kept:
        assert pilot.cell_pass(readings[name], sized[name]) >= 0.90, name
    # The family rule reads 0.95 or, with every kept cell at 8,000, records the shortfall.
    family = pilot.family_pass(readings, sized, kept)
    assert family >= 0.95 or all(sized[name] == pilot.LADDER[-1] for name in kept)
    assert family == pytest.approx(0.933, abs=5e-4)
    for name in study.PREDICTED_SHORTFALL:
        assert sized[name] == pilot.LADDER[-1]
        assert pilot.cell_pass(readings[name], pilot.LADDER[-1]) < 0.90


def test_the_predicted_reds_are_the_declared_ones(readings) -> None:
    """Cells whose pass probability at the pilot's own readings is below 0.90."""
    sized = study.PROPERTY_REPLICATES
    below = {name for name, r in readings.items() if pilot.plug_in_pass(r, sized[name]) < 0.90}
    assert below == set(study.PREDICTED_RED)


def test_the_deflated_control_has_power_where_the_variance_is_calibrated(readings) -> None:
    """At a calibrated variance the deflated arm reads 0.7225: its upper end falls below 0.8649."""
    for name, reading in readings.items():
        if name in study.PREDICTED_SHORTFALL:
            continue
        power = pilot.below_band_power(
            reading["relvar"], reading["kurtosis"], study.PROPERTY_REPLICATES[name], study.DEFLATION
        )
        assert power >= 0.95, name


def test_the_iid_control_predicted_reds_are_the_declared_ones(rows: pd.DataFrame) -> None:
    below = {
        cell.name
        for cell in study.CELLS
        if pilot.iid_power(rows, cell.name, study.PROPERTY_REPLICATES[cell.name]) < 0.90
    }
    assert below == set(study.IID_PREDICTED_RED)


def test_the_pooled_centred_gate_decision(readings) -> None:
    """No cell gates ``pooled_centered``: re-centring adds no excess to the fold variance.

    The pilot's ratio of the pooled-centred to the fold variance ratio stays below 1.25 in every
    cell (plan trigger), including the cells with 2 clusters per fold.
    """
    assert frozenset() == study.POOLED_CENTERED_GATED
    for name, reading in readings.items():
        assert reading["pooled_ratio"] / reading["ratio"] < study.POOLED_CENTERED_PILOT_TRIGGER, (
            name
        )


def test_the_closed_form_sizing_matches_the_plan_s_simulation() -> None:
    """Witness: the closed form against ``variance_ratio_sim`` (plan rev 2, k = 2)."""
    for kurt_se, kurt_psi, replicates in ((0.5, 0.0, 4_000), (2.0, 1.0, 4_000), (2.0, 1.0, 8_000)):
        relvar = 2.0 / 2.0 + kurt_se
        closed = pilot.variance_ratio_pass(relvar, kurt_psi, replicates)
        simulated = pilot.variance_ratio_sim(2, kurt_se, replicates, kurt_psi=kurt_psi, sims=400)
        assert closed == pytest.approx(simulated, abs=0.05)
    # Mutation control: an interval that ignored the spread of the squared SE would pass the
    # heavy-tailed case far more often than the simulation does.
    assert pilot.variance_ratio_pass(0.0, 0.0, 4_000) > 0.99


# ------------------------------------------------------------------------------ the rules

#: One cell keeps the constructed rows small; every rule reads each cell on its own.
RULE_CELL = "cluster_covariate__v2_j8"
POSITIVE = study.cell_name(RULE_CELL, "t_reference")
DEFLATED = study.cell_name(RULE_CELL, "deflated_control")
IID = study.cell_name(RULE_CELL, "iid_t_control")
KNOWN = study.cell_name(RULE_CELL, "known_propensity")


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
        se = {"iid_t_control": 0.6, "deflated_control": 0.85}.get(arm, 1.0)
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
                    "std_error": se * rng.uniform(0.98, 1.02, size=replicates),
                    "covered": covered,
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


def test_a_variance_above_the_band_turns_the_positive_red(passing: pd.DataFrame) -> None:
    assert not _verdicts(_scaled(passing, POSITIVE, 1.2)).loc[POSITIVE, "passed"]


def test_a_variance_below_the_band_turns_the_positive_red_with_an_se_ratio_inside(
    passing: pd.DataFrame,
) -> None:
    """Mutation control: a variance ratio of 0.76 reads an SE ratio of 0.87, inside 0.8 to 1.2."""
    verdicts = _verdicts(_scaled(passing, POSITIVE, 0.87))
    assert 0.8 < verdicts.loc[POSITIVE, "shown_se_ratio"] < 1.2
    assert verdicts.loc[POSITIVE, "variance_ratio_ci_upper"] < properties.VARIANCE_BAND[0]
    assert not verdicts.loc[POSITIVE, "passed"]


def test_each_control_fails_when_its_failure_is_absent(passing: pd.DataFrame) -> None:
    """Over-coverage reddens the positive, an undeflated arm its control; a diagnostic stays."""
    rows = _scaled(_scaled(passing, DEFLATED, 1 / 0.85), KNOWN, 3.0)
    rows.loc[rows["cell"] == POSITIVE, "covered"] = 1
    verdicts = _verdicts(rows)
    assert not verdicts.loc[POSITIVE, "passed"]
    assert not verdicts.loc[DEFLATED, "passed"]
    assert verdicts.loc[KNOWN, "passed"]
    assert verdicts.loc[IID, "passed"]


def test_an_iid_se_like_the_cluster_se_fails_its_control(passing: pd.DataFrame) -> None:
    verdicts = _verdicts(_scaled(passing, IID, 1 / 0.6))
    assert not verdicts.loc[IID, "passed"]


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


def test_the_pilot_file_is_committed() -> None:
    assert Path(PILOT).exists()
