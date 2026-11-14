"""The declaration of the ``mtp-policy-variance`` study, before its run.

Every design number is computed here from the modules, the committed tail-size CSVs, the
committed probe-5 draws and the committed pilot, not copied: each twin's n (check 2, with its
mutation control), the coverage-floor factor, the variance-learner screen's choice, each fit
set's replication count under the sizing rule, each control's power and the family-wise pass
probability.  The run itself is not started here.
"""

from __future__ import annotations

import functools
import re
import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest

from cleverly.longitudinal import policy_variance as pv
from tests.studies import canonical_mtp_boundaries as boundaries
from tests.studies import canonical_mtp_policy_variance as study
from tests.studies import canonical_mtp_pricing as pricing
from tests.studies import mtp_boundaries_properties as boundaries_properties
from tests.studies import mtp_policy_variance_properties as properties
from tests.studies import mtp_pricing_properties as pricing_properties
from tests.studies.evidence import property_failures as pf
from tests.studies.evidence.seeds import stream_seed
from tests.unit import test_mtp_tail_size as tail_rule

# ------------------------------------------------------------------------ check 2


def source_twin(name: str) -> tuple[str, str] | None:
    """The twin and the estimand that gate a source study's below-rule cell."""
    scenario, _, estimand = name.partition("/")
    if scenario == "interval_calibration":
        label = estimand.removesuffix("__correctly_specified")
        twin = f"{label}_twin"
        if twin in properties.GATED_TWINS:
            config = properties.TWIN_SETS[twin][0]
            return twin, config.names[0 if config.kind == "pricing_property" else 1]
        return None
    match = re.fullmatch(r"(pricing_(?:ltmle|sdr)_tau\d)", scenario)
    if match:
        return f"{scenario}_tail", estimand
    match = re.fullmatch(r"(atom|boundaries)_(ltmle|sdr)", scenario)
    if match:
        prefix = "atom" if match.group(1) == "atom" else "p4"
        return f"{prefix}_{match.group(2)}_tail", estimand
    return None


def declared_sizes() -> dict[str, int]:
    return {
        **{twin: study.SCENARIO_N[twin] for twin in study.TWINS},
        **{label: n for label, (_, n) in properties.TWIN_SETS.items()},
    }


def check_two(sizes: Mapping[str, int]) -> list[str]:
    """What breaks check 2: a below-rule exact cell without an in-regime gated twin."""
    cells = {cell.name: cell for cell in tail_rule.boundaries_cells() + tail_rule.pricing_cells()}
    below = (
        set(boundaries.BELOW_TAIL_SIZE)
        | set(pricing.BELOW_TAIL_SIZE)
        | set(boundaries_properties.BELOW_TAIL_SIZE)
        | set(pricing_properties.BELOW_TAIL_SIZE)
    )
    problems = []
    for name in sorted(below):
        cell = cells[name]
        twinned = source_twin(name)
        if not cell.exact:
            if name not in LOWER_BOUND_CELLS:
                problems.append(f"{name}: a lower-bound cell with no declared reason")
            continue
        if twinned is None:
            problems.append(f"{name}: no gated twin")
            continue
        twin, estimand = twinned
        gated = twin in properties.GATED_TWINS or (
            twin in study.TWINS and estimand in study.TWINS[twin][1]
        )
        if not gated:
            problems.append(f"{name}: twin {twin} does not gate {estimand}")
        need = study.gated_size(cell.n_kappa)
        if sizes[twin] < need:
            problems.append(f"{name}: twin {twin} at {sizes[twin]} below {need}")
    return problems


#: The below-rule cells with only a lower bound of their tail size, and what they get.
LOWER_BOUND_CELLS = {
    "interval_calibration/up_learned_classifier_tau5__correctly_specified": (
        "reporting twin up_learned_classifier_tau5_twin at 4,000"
    ),
    "interval_calibration/up_learned_density_tau5__correctly_specified": (
        "none: a conservative red cell under its own owner"
    ),
}


def test_every_below_rule_cell_has_an_in_regime_gated_twin() -> None:
    assert check_two(declared_sizes()) == []


@pytest.mark.parametrize("twin", ["pricing_ltmle_tau3_tail", "atom_sdr_tail", "p5_twin"])
def test_check_two_fails_when_a_twin_is_400_rows_short(twin: str) -> None:
    sizes = declared_sizes()
    sizes[twin] -= study.STEP
    assert any(twin in problem for problem in check_two(sizes))


def test_the_twin_sizes_are_the_declared_table() -> None:
    assert {
        "pricing_ltmle_tau5_tail": 72_800,
        "pricing_sdr_tau5_tail": 73_600,
        "pricing_ltmle_tau3_tail": 5_600,
        "pricing_sdr_tau3_tail": 5_600,
        "atom_ltmle_tail": 6_400,
        "atom_sdr_tail": 6_400,
        "p4_ltmle_tail": 104_400,
        "p4_sdr_tail": 104_400,
        **{study.policy_scenario(t): study.SCENARIO_N[t] for t in study.OPTION_TWINS},
    } == study.SCENARIO_N
    assert {label: n for label, (_, n) in properties.TWIN_SETS.items()} == {
        "p4_twin": 104_400,
        "p4_sdr_twin": 104_400,
        "p5_twin": 6_400,
        "p5_sdr_twin": 6_400,
        "up_oracle_tau5_twin": 4_000,
        "up_learned_classifier_tau5_twin": 4_000,
    }


# ------------------------------------------------------------------------ the record


def test_the_record_publishes_the_cells_the_fit_sets_write() -> None:
    declared = properties.declared_cell_names()
    assert set(study.PROPERTY_CELLS) == set(declared)
    for family, cells in study.PROPERTY_CELLS.items():
        assert set(cells) == declared[family], family
    assert study.STUDY.property_cells == study.PROPERTY_CELLS


def test_every_scored_family_has_a_rule_and_every_reporting_family_is_diagnostic() -> None:
    ruled = {family for rule in properties.PROPERTY_RULES for family in rule.families}
    roles = {
        family: {properties.own_cells(family, label).get(cell) for cell in cells}
        for family, label, _ in properties.FIT_SETS
        for cells in [properties.own_cells(family, label)]
    }
    for family, role_set in roles.items():
        if role_set - {properties.DIAGNOSTIC_ROLE}:
            assert family in ruled, family


def test_the_record_is_gated_and_reports_the_twins_option_rows() -> None:
    record = study.STUDY
    assert record.publication_policy == "gated"
    assert record.reference is None
    assert record.reported_scenarios == frozenset(
        f"{t}_policy" for t in ("pricing_ltmle_tau3_tail", "pricing_sdr_tau3_tail")
    ) | frozenset(f"{t}_policy" for t in ("atom_ltmle_tail", "atom_sdr_tail"))
    assert study.CONFIGURATION["variance_learner"] == dict(pv.VARIANCE_LEARNER)


# ------------------------------------------------------------------------ the screen


@functools.cache
def screen_summary() -> pd.DataFrame:
    return study.screen_summary(study.committed_screen())


def test_the_screen_ran_every_held_out_draw() -> None:
    rows = study.committed_screen()
    counts = rows.groupby(["config", "estimator"])["replicate"].nunique()
    assert set(counts.index.get_level_values(0)) == set(study.SCREEN_CONFIGS)
    assert (counts == study.SCREEN_DRAWS).all()


def test_the_screen_choice_is_declared_and_shipped() -> None:
    choice = study.screen_choice(screen_summary())
    assert choice == study.SCREEN_CHOICE
    shipped = {key: pv.VARIANCE_LEARNER[key] for key in study.CANDIDATES[choice]}
    assert shipped == study.CANDIDATES[choice]


def test_the_screen_rule_sees_a_cell_outside_the_band() -> None:
    """Mutation control: a d5 cell pushed to 1.26 drops d5, and the rule then reads d3."""
    summary = screen_summary().copy()
    summary.loc[summary.index[0], "d5_se_ratio"] = 1.26
    in_band = summary["d3_se_ratio"].between(*study.SCREEN_BAND).all()
    assert study.screen_choice(summary) == ("d3" if in_band else "none")


def test_the_screen_reads_the_other_candidate_as_a_separate_fit() -> None:
    """One held-out draw: the d3 SE beside the d5 fit equals a fit under d3."""
    config = study.SCREEN_CONFIGS["lawP:price_down:3"][0]
    frame = config.sample(2_000, stream_seed(study.STUDY, "diagnostic", "screen-check"))
    store: dict[str, list] = {}
    with study.quiet(), study.both_candidates(store):
        result = config.fit(frame, study.OPTION)
    name = config.names[1]
    beside = study._candidate_se(result[name], store["d5"][0], store["d3"][0])
    shipped = pv.VARIANCE_LEARNER
    pv.VARIANCE_LEARNER = {**shipped, **study.CANDIDATES["d3"]}  # type: ignore[misc]
    try:
        with study.quiet():
            alone = study.readings(config.fit(frame, study.OPTION), (name,))[name]["se_pw"]
    finally:
        pv.VARIANCE_LEARNER = shipped  # type: ignore[misc]
    assert beside == pytest.approx(alone, rel=1e-9)
    assert beside != pytest.approx(study.readings(result, (name,))[name]["se_pw"], rel=1e-6), (
        "the two candidates must differ on this draw"
    )


# ------------------------------------------------------------------------ the controls


def test_the_coverage_floor_factor_is_the_largest_declared_one_with_power() -> None:
    """Rev 4.1's rule at the declared counts: the largest factor with power on every row."""
    assert properties.chosen_floor_factor() == properties.COVERAGE_FLOOR_FACTOR
    powers = properties.floor_control_power(properties.COVERAGE_FLOOR_FACTOR)
    assert powers["power"].min() >= properties.CONTROL_POWER
    assert set(powers["set"]) == {
        label
        for label in (*properties.OPTION_SETS, *properties.CALIBRATION_SETS)
        if label not in properties.REPORTING_SETS and "tau2" not in label
    }


def test_the_next_larger_factor_lacks_power() -> None:
    """The control of the choice: one step up misses 0.95 on some probed row."""
    index = properties.FLOOR_FACTORS.index(properties.COVERAGE_FLOOR_FACTOR)
    larger = properties.FLOOR_FACTORS[index - 1]
    assert properties.floor_control_power(larger)["power"].min() < properties.CONTROL_POWER


@pytest.mark.parametrize("label", sorted(properties.WALD_CONTROL_SETS))
def test_the_wald_control_fails_on_the_committed_draws(label: str) -> None:
    """The below-tail control, Jeffreys over the 1,000 committed draws alone."""
    for q in properties.QUANTITIES:
        covered, draws = properties.wald_control_draws(label, f"{label}__{q}__wald")
        power = properties.jeffreys_below(
            covered, draws, properties.REPLICATES[label], properties.WALD_CONTROL_CEILING
        )
        assert power >= properties.CONTROL_POWER, (label, q, power)


# ------------------------------------------------------------------------ the pilot and sizing


@functools.cache
def sizing() -> tuple[dict[str, int], int, pd.DataFrame]:
    return properties.sized_replicates(properties.committed_pilot())


def test_the_pilot_ran_every_declared_set_and_round_two() -> None:
    pilot = properties.committed_pilot()
    counts = pilot.groupby("set")["replicate"].agg(["nunique", "max"])
    expected = {label for _, label, _, _ in properties.PILOT_SETS}
    assert set(counts.index) == expected
    for label, row in counts.iterrows():
        draws = properties.pilot_draws(str(label))
        assert row["nunique"] == draws and row["max"] == draws - 1, label


def test_round_two_extends_the_rows_rule_s3_flags() -> None:
    """S3 on the round-1 pilot: a plug-in pass probability below 0.90 at ``R_target``."""
    assert properties.round_two_sets(properties.committed_pilot()) == properties.ROUND_TWO


def test_rule_s3_flags_a_row_moved_off_target() -> None:
    """Mutation control: a twin's round-1 ratio pushed to 1.05 is flagged."""
    pilot = properties.round_one(properties.committed_pilot()).copy()
    rows = pilot["set"] == "up_oracle_tau5_twin"
    pilot.loc[rows, "std_error"] *= 1.05 / 1.0075
    assert "up_oracle_tau5_twin" in properties.round_two_sets(pilot)


def test_no_round_two_row_is_confirmed_outside_its_passing_region() -> None:
    """S3: a departure the extended pilot confirms stops the declaration for a diagnosis."""
    _, _, table = sizing()
    extended = table.loc[table["set"].isin(properties.ROUND_TWO)]
    assert set(extended["set"]) == set(properties.ROUND_TWO)
    for row in extended.itertuples():
        assert row.draws == properties.ROUND_TWO[row.set], row.cell
        assert properties.in_region(row.kind, row.se_ratio, row.coverage, row.standardized_bias), (
            row.cell,
            row.se_ratio,
            row.coverage,
            row.standardized_bias,
        )


def test_the_gated_rows_and_their_per_row_level() -> None:
    _, _, table = sizing()
    assert len(table) == 33
    assert table["level"].iloc[0] == pytest.approx(0.95 ** (1 / 33))
    assert table["level"].iloc[0] == pytest.approx(0.9985, abs=1e-4)
    kinds = table.groupby("kind").size().to_dict()
    assert kinds == {"band": 5, "clause": 14, "primary": 14}


def test_the_configuration_constant_floors_at_c_fit() -> None:
    _, _, table = sizing()
    assert (table["c"] >= properties.tools.C_FIT - 1e-12).all()
    c = table.groupby("configuration")["c"].nunique()
    assert (c == 1).all()
    assert table.loc[table["configuration"] == "pricing_tau5", "c"].iloc[0] == pytest.approx(
        1.35, abs=0.01
    )


def test_r_target_is_the_least_count_at_the_per_row_level() -> None:
    """S1: at ``R_target`` the at-target probability reaches the level; one step below does not."""
    _, _, table = sizing()
    for row in table.itertuples():
        assert properties.at_target(row.kind, row.r_target, row.c) >= row.level, row.cell
        floor = properties.PRIMARY_FLOOR if row.kind == "primary" else properties.R_STEP
        if row.r_target > floor:
            below = row.r_target - properties.R_STEP
            assert properties.at_target(row.kind, below, row.c) < row.level, row.cell


def test_the_plugin_raise_follows_rule_s2() -> None:
    _, _, table = sizing()
    for row in table.itertuples():
        point = (row.kind, row.se_ratio, row.coverage, row.standardized_bias)
        if not row.margin:
            assert row.r_plugin == row.r_target, row.cell
            continue
        assert row.r_target <= row.r_plugin <= max(properties.S2_CAP, row.r_target), row.cell
        reached = properties.plugin_pass(*point, row.r_plugin, row.c) >= properties.S2_LEVEL
        assert reached or row.r_plugin == properties.S2_CAP, row.cell
        if row.r_plugin > row.r_target:
            below = row.r_plugin - properties.R_STEP
            assert properties.plugin_pass(*point, below, row.c) < properties.S2_LEVEL, row.cell


def test_the_replication_counts_are_the_sizing_rule() -> None:
    replicates, primary, _ = sizing()
    assert primary == study.PRIMARY_REPLICATES
    assert primary >= properties.PRIMARY_FLOOR
    for label, count in replicates.items():
        assert properties.REPLICATES[label] == count, label


def test_the_replication_counts_of_the_reporting_sets() -> None:
    assert (
        properties.REPLICATES["up_learned_classifier_tau5_twin"]
        == (properties.REPLICATES["up_oracle_tau5_twin"])
    )
    for label in properties.REPORTING_SETS:
        assert properties.REPLICATES[label] == 1_000
    assert properties.REPLICATES[properties.SUPERLEARNER_CELL] == properties.trigger_replicates()
    for twin in study.LARGE_TWINS:
        assert properties.REPLICATES[twin] == study.TAIL_OPTION_REPLICATES


def test_the_follow_up_trigger_detects_its_declared_shortfall() -> None:
    """The default-pseudo-learner cell: 0.92 detected with probability 0.95, not one step less."""
    count = properties.trigger_replicates()
    assert count == 1_200
    assert properties.trigger_power(properties.TRIGGER_DETECTS, count) >= properties.CONTROL_POWER
    below = count - properties.R_STEP
    assert properties.trigger_power(properties.TRIGGER_DETECTS, below) < properties.CONTROL_POWER


def test_the_nominal_clause_rides_on_the_four_tau5_option_rows() -> None:
    _, _, table = sizing()
    nominal = table.loc[table["nominal"], "cell"]
    assert sorted(nominal) == sorted(
        f"{label}__{q}__option" for label in properties.NOMINAL_SETS for q in properties.QUANTITIES
    )
    # At the target the clause holds with about its one-sided level, at any count.
    for r in (2_000, 6_000, 20_000):
        assert 0.993 <= properties.nominal_pass(0.95, r) <= 0.997
    # The default it replaces fails it: the committed Wald draws cover 0.916 to 0.923.
    for label in properties.NOMINAL_SETS:
        for q in properties.QUANTITIES:
            covered, draws = properties.wald_control_draws(label, f"{label}__{q}__wald")
            assert properties.nominal_pass(covered / draws, 6_000) < 0.01


def test_the_at_target_product_reaches_the_target() -> None:
    """The gate: every gated row at its target reads a false red with probability <= 0.05."""
    replicates, primary, table = sizing()
    product = properties.at_target_product(table, replicates, primary)
    assert product >= properties.FAMILY_TARGET
    assert product == pytest.approx(properties.PUBLISHED["at_target"], abs=5e-4)
    targets = table.groupby("set")["r_target"].max().to_dict()
    at_r_target = properties.at_target_product(
        table, targets, int(table.loc[table["kind"] == "primary", "r_target"].max())
    )
    assert at_r_target >= properties.FAMILY_TARGET
    assert at_r_target == pytest.approx(properties.PUBLISHED["at_r_target"], abs=5e-4)


def test_the_published_descriptions() -> None:
    """The plug-in and predictive products gate nothing; their published values are these."""
    replicates, primary, table = sizing()
    plugin = properties.plugin_product(table, replicates, primary)
    predictive = properties.family_wise(
        properties.pass_table(properties.committed_pilot(), properties.REPLICATES, primary)
    )
    assert plugin == pytest.approx(properties.PUBLISHED["plug_in"], rel=0.05, abs=5e-4)
    assert predictive == pytest.approx(properties.PUBLISHED["predictive"], rel=0.05, abs=5e-4)


def test_every_control_has_power() -> None:
    """S4: each control fails as required with probability >= 0.95, predictive over its pilot."""
    table = properties.pass_table(
        properties.committed_pilot(), properties.REPLICATES, study.PRIMARY_REPLICATES
    )
    controls = table.loc[table["role"] == "control"]
    assert len(controls) > 0
    weak = controls.loc[controls["probability"] < properties.CONTROL_POWER]
    assert weak.empty, weak.to_string()


def test_the_twin_floor_factor_is_the_largest_declared_one_with_power() -> None:
    pilot = properties.committed_pilot()
    primary = study.PRIMARY_REPLICATES
    assert properties.chosen_twin_floor_factor(pilot, primary) == study.CONTROL_SE_FACTOR
    index = properties.FLOOR_FACTORS.index(study.CONTROL_SE_FACTOR)
    larger = properties.FLOOR_FACTORS[index - 1]
    assert properties.twin_floor_power(larger, pilot, primary) < properties.CONTROL_POWER


def test_the_two_families_declare_their_own_floor_factor() -> None:
    assert np.isclose(study.CONTROL_SE_FACTOR, 0.65)
    assert np.isclose(properties.COVERAGE_FLOOR_FACTOR, 0.60)


def test_the_rules_reproduce_the_verdicts_on_the_pilot() -> None:
    """The accounting rules read the same verdicts as the study's own code."""
    pilot = properties.round_one(properties.committed_pilot())
    chosen = pilot.loc[pilot["set"].isin(["pricing_ltmle_tau5", "p4_ltmle", "up_oracle_tau5_twin"])]
    rows = chosen.loc[:, list(properties.REPLICATE_COLUMNS)]
    rows = pd.concat([rows, properties.derived_rows(rows)], ignore_index=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = properties.summarize_properties(rows)
    ctx = pf.Context(study.STUDY, rows, summary)
    for rule in properties.PROPERTY_RULES:
        verdicts = pf.rule_verdicts(rule, summary, ctx)
        joined = verdicts.merge(
            summary[["property", "cell", "passed"]], on=["property", "cell"], suffixes=("_rule", "")
        )
        assert len(joined) == len(verdicts) > 0
        assert (joined["passed_rule"] == joined["passed"]).all(), joined.to_string()


def _pilot_primary() -> pd.DataFrame:
    pilot = properties.committed_pilot()
    rows = pilot.loc[pilot["property"] == "primary"].copy()
    parts = rows["cell"].str.split("__", n=1, expand=True)
    rows["scenario"], rows["estimand"] = parts[0], parts[1]
    return rows


def test_the_regime_contrast_reads_the_twin_against_its_control() -> None:
    pilot = properties.committed_pilot()
    contrast = study.regime_contrast(_pilot_primary(), pilot)
    assert len(contrast) == 4
    assert (contrast["twin_replicates"] == properties.PILOT_DRAWS).all()
    assert (contrast["control_replicates"] == properties.PILOT_DRAWS).all()
    assert (contrast["ci_lower"] < contrast["difference"]).all()
    assert (contrast["difference"] < contrast["ci_upper"]).all()
    # Mutation control: a twin that covers every draw against a control at 0.92 reads a gain.
    primary = _pilot_primary()
    primary["std_error"] *= 100.0
    control = pilot.copy()
    wald = control["cell"].str.endswith("__wald") & control["set"].isin(properties.NOMINAL_SETS)
    control.loc[wald, "covered"] = (control.loc[wald, "replicate"] % 25 >= 2).astype(int)
    assert (study.regime_contrast(primary, control)["reading"] == "gain in regime").all()


def test_the_newcombe_interval_matches_a_known_value() -> None:
    """Newcombe (1998), table II, example (a): 56/70 - 48/80, 95% interval (0.0524, 0.3339)."""
    low, high = study.newcombe(56, 70, 48, 80, level=0.95)
    assert low == pytest.approx(0.0524, abs=1e-4)
    assert high == pytest.approx(0.3339, abs=1e-4)


def test_the_code_read_labels() -> None:
    group = pd.DataFrame({"std_error": np.arange(1.0, 9.0), "covered": [0, 1, 1, 1, 1, 1, 1, 1]})
    point, low, high = properties.q1_coverage(group)
    assert point == pytest.approx(0.5) and low < 0.5 < high
    assert properties.rms_label(0.97, 1.02).startswith("rms: contains 1")
    assert properties.rms_label(0.90, 0.98).startswith("rms: below 1")
    assert properties.learned_label(0.96, 1.3) == "learned: as declared"
    assert properties.learned_label(0.93, 1.3).startswith("learned: departs")
    assert properties.red_route(0.98, 1.08, 0.94, 0.97) == "finite-sample candidate"
    assert properties.red_route(0.90, 0.99, 0.94, 0.97) == "defect suspect"
    assert properties.red_route(0.98, 1.08, 0.90, 0.94) == "defect suspect"


def test_the_reading_names_each_clause_branch() -> None:
    assert properties.reading(0.92, 1.0, True, True) == "red: too narrow"
    assert properties.reading(1.26, 1.4, True, True) == "red: too wide"
    assert properties.reading(1.0, 1.08, True, True) == "conservative"
    assert properties.reading(0.95, 1.07, True, True) == "calibrated"
    assert properties.reading(0.95, 1.07, False, True) == "red: coverage"
    assert properties.reading(0.95, 1.07, True, True, False) == "red: below nominal"
    assert np.isclose(properties.CLAUSE_FLOOR, 0.93) and np.isclose(properties.CLAUSE_CEILING, 1.25)
