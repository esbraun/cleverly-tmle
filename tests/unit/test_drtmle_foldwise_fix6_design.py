"""Design of the three gated Fix-6 records of the fold-wise DR-TMLE default.

``drtmle-foldwise-n1500`` (unit B), ``multi-arm-drtmle-foldwise`` (unit M) and
``stratified-drtmle-foldwise`` (unit S, whose contraction ladder is its property cells).  Each
test names the part of the declaration it pins: the record, the pairing of its draws with the
registered study they re-read, the fresh streams, the declared budgets, which each test
recomputes from the committed pilot summaries (``tests/studies/drtmle_foldwise_pilot.json``)
with the record's sizing rule, the acceptance texts declared verbatim, the guard, and, for unit
B, the nonzero witness that its fold-wise path is not wired pooled.  No test runs a study; the
witness and the pilot check fit a few samples.
"""

from __future__ import annotations

import inspect
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from cleverly.validation._seeds import sample_seed_streams
from tests.diagnostics.fix6_foldwise_sizing import run as pilot
from tests.studies import canonical_drtmle, drtmle_properties, stratified_alternating_law
from tests.studies import canonical_multi_arm_drtmle as canonical_multi_arm
from tests.studies import canonical_stratified_drtmle as canonical_stratified
from tests.studies import drtmle_foldwise as foldwise
from tests.studies import drtmle_foldwise_n1500 as unit_b
from tests.studies import multi_arm_drtmle_foldwise as unit_m
from tests.studies import multi_arm_drtmle_properties as registered_multi_arm
from tests.studies import stratified_drtmle_foldwise as unit_s
from tests.studies import stratified_drtmle_foldwise_ladder as ladder
from tests.studies import stratified_drtmle_properties as x8
from tests.studies.drtmle_foldwise_sizing import (
    PLANNING_PATH,
    SET_ASIDE_THRESHOLD,
    LadderPlan,
    Pilot,
    Rule,
    committed_planning,
    pilot_from_rows,
    predictive_outer_budget,
    set_aside_rows,
    slope_half_width,
)
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.registry import ROOT, StudyRecord, registered
from tests.studies.evidence.seeds import replicate_seed, stream_seed

MODULES: dict[str, ModuleType] = {
    "drtmle-foldwise-n1500": unit_b,
    "multi-arm-drtmle-foldwise": unit_m,
    "stratified-drtmle-foldwise": unit_s,
}
#: Each record's properties module.
PROPERTIES: dict[str, ModuleType] = {
    "drtmle-foldwise-n1500": unit_b,
    "multi-arm-drtmle-foldwise": unit_m,
    "stratified-drtmle-foldwise": ladder,
}
DIRECTORIES = {
    "drtmle-foldwise-n1500": "drtmle_foldwise_n1500",
    "multi-arm-drtmle-foldwise": "multi_arm_drtmle_foldwise",
    "stratified-drtmle-foldwise": "stratified_drtmle_foldwise",
}

#: The smallest |fold-wise - pooled| difference the unit B witness accepts on its draw.  The
#: draw reads 0.026; float drift between two pooled fits of one draw is about 1e-16.
FOLDWISE_WITNESS_FLOOR = 1e-3
#: How closely the pooled refit reproduces the committed RM18 estimate.
POOLED_REPRODUCTION_TOLERANCE = 1e-10
#: How closely a recomputed predictive pass must match the declared one.  The rule's draws are
#: seeded, so the value moves only with library rounding.
PASS_TOLERANCE = 0.002

#: The ``RM18-one-sided-bias`` acceptance as declared: Fix-6's D2 text with the closing clause
#: of the R-a ruling.
RM18_ACCEPTANCE = (
    'Standing record of `reduced_crossfit="pooled"`, the R `drtmle` `cvFolds` construction. '
    "Corrected on the default: the foldwise reduced regressions (X83) pass on the binary law "
    "for `treatment_correct` at n = 1,000 and for both regimes at n = 3,000 "
    "(`drtmle-foldwise`), and on the multi-arm law at n = 2,000 on this cell's draws "
    "(`drtmle-foldwise`). The owner closes when both paired cells of `drtmle-foldwise-n1500`, "
    "the default on the binary cells' own draws at n = 1,500, pass, with the record's primary "
    "rows passing; a red cell there is a defect suspect and the owner does not close. The "
    "three cells keep their verdicts."
)
#: The ``X8-drtmle-one-sided-bias`` acceptance as declared: Fix-6's D4 text, unchanged.
X8_ACCEPTANCE = (
    'The four cells fit `reduced_crossfit="pooled"` and keep their verdicts. The owner closes '
    "as a finite-sample limit of the default when, in `stratified-drtmle-foldwise`, the "
    "99% slope interval of log |bias| on log n has its upper end below -1/2 for the marginal "
    "ATE and for `v1` (each read whatever its realised half-width; a half-width above 0.25 for "
    "either reads `not resolved`, and the owner does not close), the `both_wrong` control "
    "fails to contract, and no rung's SE-ratio or coverage row is a defect suspect under the "
    "finite-sample rule. Otherwise the red is a defect suspect and is diagnosed, and this "
    "owner does not close."
)


def _record(slug: str) -> StudyRecord:
    (record,) = (study for study in registered() if study.slug == slug)
    return record


# ------------------------------------------------------------------------------ every record
@pytest.mark.parametrize("slug", sorted(MODULES))
def test_the_record_registers_gated_with_property_cells_and_no_comparator(slug: str) -> None:
    module = MODULES[slug]
    properties = PROPERTIES[slug]
    record = _record(slug)
    assert record is module.STUDY
    assert record.reference is None
    assert record.pending_reference is None
    assert record.comparator_state == "none"
    assert record.publication_policy == "gated"
    assert record.property_cells
    assert not record.reported_scenarios
    assert record.implementation == foldwise.IMPLEMENTATION
    assert dict(record.failure_bound) == {foldwise.IMPLEMENTATION: 0.01}
    assert properties.FAILURE_BOUND == 0.01
    assert properties.PROPERTY_WITNESS is property_verdicts.SAMPLE_WITNESS
    assert record.artifacts == ROOT / "tests" / "canonical" / DIRECTORIES[slug]
    assert (record.artifacts / "__init__.py").is_file()
    assert (record.artifacts / "regenerate.py").is_file()
    assert record.runner() is module
    assert record.properties() is properties
    assert not hasattr(module, "PRIMARY_DESIGN_OPEN")
    assert record.replicates == module.PRIMARY_REPLICATES
    assert module.CONFIGURATION["reduced_crossfit"] == "foldwise"


def test_the_resampling_seeds_are_fresh_and_distinct() -> None:
    ours = {slug: _record(slug) for slug in MODULES}
    others = [study for study in registered() if study.slug not in MODULES]
    taken = {study.seed for study in others} | {
        study.resampling_seed for study in others if study.resampling_seed is not None
    }
    resampling = [record.resampling_seed for record in ours.values()]
    assert len(set(resampling)) == len(resampling)
    assert not set(resampling) & taken
    # Unit B's primary draws a seed no other study uses.
    assert ours["drtmle-foldwise-n1500"].seed not in taken
    assert not any(study.slug == "stratified-drtmle-foldwise-ladder" for study in registered())
    # The two paired primaries share their canonical study's seed on purpose.
    assert unit_m.SEED == canonical_multi_arm.SEED
    assert unit_s.SEED == canonical_stratified.SEED


@pytest.mark.parametrize("slug", sorted(MODULES))
def test_the_budgets_are_declared_and_the_guard_refuses_planning_values(
    slug: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = MODULES[slug]
    assert module.BUDGETS_DECLARED is True
    module.assert_declared()
    monkeypatch.setattr(module, "BUDGETS_DECLARED", False)
    with pytest.raises(RuntimeError, match="planning"):
        module.assert_declared()
    # The guard runs before any draw, in the primary phase and in the property phase.
    with pytest.raises(RuntimeError, match="planning"):
        module.draw_and_fit(replicates=module.PRIMARY_REPLICATES, n=40, n_jobs=1)
    with pytest.raises(RuntimeError, match="planning"):
        PROPERTIES[slug].generate_property_rows(n_jobs=1)


def test_the_committed_pilot_summaries_are_the_merged_sizing_pilot() -> None:
    """Each group holds the merged pilot: 1,000 draws per primary or cell group, 600 per rung."""
    assert PLANNING_PATH == ROOT / "tests" / "studies" / "drtmle_foldwise_pilot.json"
    plans, seconds = committed_planning()
    assert set(plans) == set(seconds) == set(pilot.GROUPS)
    for name, plan in plans.items():
        rungs = name.startswith("ladder_")
        assert isinstance(plan, LadderPlan if rungs else Pilot), name
        count = 600 if rungs else 1_000
        assert plan.replicates == count, name
        assert plan.source == f"Fix-6 sizing pilot, {name}, {count:,} replications"
        assert seconds[name] > 0


@pytest.mark.parametrize("slug", sorted(MODULES))
def test_each_record_states_its_cost_from_the_pilot_seconds(slug: str) -> None:
    module = MODULES[slug]
    assert abs(module.CPU_HOURS - module.cpu_hours()) <= 0.05
    if slug == "stratified-drtmle-foldwise":
        assert abs(ladder.CPU_HOURS - ladder.cpu_hours()) <= 0.05


@pytest.mark.parametrize("rule", [unit_b.rule, unit_m.rule, unit_s.rule], ids=["B", "M", "S"])
def test_no_row_is_set_aside(rule: Any) -> None:
    """A coverage or SE-ratio row no acceptance reads is set aside below 0.80 at the ceiling."""
    assert SET_ASIDE_THRESHOLD == 0.80
    assert set_aside_rows(rule())[1] == ()


# ---------------------------------------------------------------------------------- unit B
def test_unit_b_refits_the_two_rm18_cells_at_their_size_and_budget() -> None:
    record = unit_b.STUDY
    assert record.property_cells == {"double_robustness": ("outcome_correct", "treatment_correct")}
    for scenario in unit_b.SCENARIOS:
        cell = unit_b.registered_cell(scenario)
        assert (cell.property, cell.cell, cell.scenario) == ("double_robustness",) + (scenario,) * 2
        assert cell.n == unit_b.PRIMARY_N == 1_500
        assert cell.replicates == unit_b.REGISTERED_REPLICATES == 800
        assert cell.role == "positive"
    assert unit_b.registered_cell("outcome_correct").seed_offset == 11_000
    assert unit_b.registered_cell("treatment_correct").seed_offset == 12_000
    assert unit_b.FLOORS == {"primary": 1_000, "cells": unit_b.REGISTERED_REPLICATES}
    assert unit_b.BUDGETS == {"primary": 2_000, "cells": 5_600}
    assert unit_b.PRIMARY_REPLICATES == 2_000
    names = {f"double_robustness/{s}" for s in unit_b.SCENARIOS}
    produced = {name for key in unit_b.payloads(2) for name in unit_b._cells_of(key)}
    assert produced == names
    assert len(unit_b.payloads(2)) == 4
    assert len(unit_b.payloads()) == 2 * unit_b.BUDGETS["cells"]


@pytest.mark.parametrize("scenario", unit_b.SCENARIOS)
def test_unit_b_draws_are_the_rm18_cells_registered_draws(scenario: str) -> None:
    cell = unit_b.registered_cell(scenario)
    for replicate in (0, 1):
        ours, truth = unit_b._draw_cell(((scenario, replicate, 2),))
        theirs, _ = drtmle_properties._draw(((cell, replicate),))
        pd.testing.assert_frame_equal(ours, theirs)
        assert truth == canonical_drtmle.truth()
    # Past the registered budget the same stream continues.
    beyond = unit_b.REGISTERED_REPLICATES
    assert unit_b.sample_seed(scenario, beyond) == stream_seed(
        canonical_drtmle.STUDY,
        "property",
        "double_robustness",
        scenario,
        str(beyond + cell.seed_offset),
    )


def test_unit_b_fits_drtmle_foldwise_primary_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cell calls X83's fit at its primary configuration, ``canonical_drtmle`` untouched."""
    calls: list[tuple[str, object]] = []

    def spy(frame: pd.DataFrame, scenario: str, fit: object) -> object:
        calls.append((scenario, fit))
        raise RuntimeError("spied")

    monkeypatch.setattr(foldwise, "fit_cleverly", spy)
    drawn = unit_b._draw_cell((("treatment_correct", 0, 2),))
    with pytest.raises(RuntimeError, match="spied"):
        unit_b._fit_key((("treatment_correct", 0, 2),), drawn)
    assert calls == [("treatment_correct", foldwise.Fit())]
    assert foldwise.PRIMARY_FIT.reduced_crossfit == "foldwise"
    signature = inspect.signature(canonical_drtmle.fit_cleverly)
    assert "reduced_crossfit" not in signature.parameters
    assert canonical_drtmle.CONFIGURATION["reduced_crossfit"] == "pooled"


def test_unit_b_foldwise_fit_is_a_nonzero_witness_against_the_registered_pooled_estimate() -> None:
    """On one registered draw the fold-wise fit moves, and the pooled refit is the RM18 row.

    A record whose fold-wise path were wired pooled would read a zero difference and fail the
    floor; a draw that were not the RM18 draw would miss the committed estimate.
    """
    scenario, replicate = "treatment_correct", 1
    frame, truth = unit_b._draw_cell(((scenario, replicate, 2),))
    rows = unit_b._fit_key(((scenario, replicate, 2),), (frame, truth))
    pooled = foldwise.fit_cleverly(frame, scenario, foldwise.Fit(reduced_crossfit="pooled"))
    committed = pd.read_parquet(canonical_drtmle.STUDY.artifact("property-replicates.parquet"))
    (registered_estimate,) = committed.loc[
        (committed["property"] == "double_robustness")
        & (committed["cell"] == scenario)
        & (committed["replicate"] == replicate),
        "estimate",
    ]
    pooled_estimate = float(pooled.estimates["ate"].psi)
    assert abs(pooled_estimate - float(registered_estimate)) <= POOLED_REPRODUCTION_TOLERANCE
    assert abs(float(rows[0]["estimate"]) - pooled_estimate) > FOLDWISE_WITNESS_FLOOR
    # The declaration reads a property row's bounds as estimate +/- z * SE.
    (row,) = rows
    z = float(norm.ppf(1 - unit_b.STUDY.margins.alpha / 2))
    half = z * float(row["std_error"])
    covered = abs(float(row["estimate"]) - float(row["truth"])) <= half
    assert int(row["covered"]) == int(covered)


def test_unit_b_rule_sizes_the_both_correct_primary_and_the_cells_jointly() -> None:
    rule = unit_b.rule()
    assert unit_b.PLANNING_GROUPS == {
        "primary_both_correct": "b_both_correct",
        "cell_outcome_correct": "b_outcome_correct",
        "cell_treatment_correct": "b_treatment_correct",
    }
    assert rule.units == {
        "primary": (("primary_both_correct",), 1_000),
        "cells": (("cell_outcome_correct", "cell_treatment_correct"), 800),
    }
    assert rule.protected == {("cell_outcome_correct", "ate"), ("cell_treatment_correct", "ate")}
    primary = rule.pilots["primary_both_correct"]
    assert primary.rows == ("ey0", "ey1", "ate") and primary.calibrated
    assert np.shape(primary.influence) == (9, 9)
    for scenario in unit_b.SCENARIOS:
        cell = rule.pilots[f"cell_{scenario}"]
        assert cell.rows == ("ate",) and not cell.calibrated
    # The treatment_correct cell sits near the margin by design.
    assert rule.pilots["cell_treatment_correct"].bias == (pytest.approx(0.1488, abs=1e-4),)
    # Two units share the family-wise target; the declared budgets are the rule's.
    assert rule.unit_target == pytest.approx(0.95**0.5)
    assert rule.budget("primary") == (2_000, pytest.approx(0.9786, abs=PASS_TOLERANCE), True)
    assert rule.budget("cells") == (5_600, pytest.approx(0.9747, abs=PASS_TOLERANCE), True)
    assert {unit: rule.budget(unit)[0] for unit in rule.units} == dict(unit_b.BUDGETS)
    assert rule.family_pass(unit_b.BUDGETS) >= 0.95


def test_unit_b_declares_the_rm18_acceptance_verbatim() -> None:
    assert unit_b.RM18_ACCEPTANCE == RM18_ACCEPTANCE


def test_unit_b_primary_is_the_gated_both_correct_regime_on_fresh_draws() -> None:
    record = unit_b.STUDY
    assert unit_b.PRIMARY_SCENARIOS == ("both_correct",)
    assert record.scenarios == {"both_correct": canonical_drtmle.ESTIMANDS}
    assert record.reported_scenarios == frozenset()
    assert record.n == unit_b.PRIMARY_N == 1_500
    cells = {unit_b.sample_seed(scenario, r) for scenario in unit_b.SCENARIOS for r in range(50)}
    primary = {replicate_seed(record, "both_correct", r) for r in range(50)}
    assert not primary & cells
    assert replicate_seed(record, "both_correct", 0) != replicate_seed(
        canonical_drtmle.STUDY, "both_correct", 0
    )
    frame, truth = unit_b.draw_scenario("both_correct", 40, 0)
    assert len(frame) == 40 and truth == canonical_drtmle.truth()


# ---------------------------------------------------------------------------------- unit M
def test_unit_m_is_the_canonical_multi_arm_primary_with_foldwise_reduced_regressions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = unit_m.STUDY
    assert record.scenarios == {
        canonical_multi_arm.SCENARIO: canonical_multi_arm.STUDY.scenarios[
            canonical_multi_arm.SCENARIO
        ]
    }
    assert record.n == canonical_multi_arm.PRIMARY_N == 2_000
    assert record.property_cells == {"double_robustness": ("outcome_correct",)}
    calls: list[tuple[str, str]] = []

    def spy(frame: pd.DataFrame, scenario: str, *, reduced_crossfit: str) -> str:
        calls.append((scenario, reduced_crossfit))
        return "fitted"

    monkeypatch.setattr(canonical_multi_arm, "fit_cleverly", spy)
    assert unit_m.fit_cleverly(pd.DataFrame()) == "fitted"
    assert calls == [(canonical_multi_arm.SCENARIO, "foldwise")]
    monkeypatch.undo()
    default = inspect.signature(canonical_multi_arm.fit_cleverly).parameters["reduced_crossfit"]
    assert default.default == canonical_multi_arm.CONFIGURATION["reduced_crossfit"] == "pooled"


def test_unit_m_draws_are_the_registered_multi_arm_draws() -> None:
    for replicate in (0, 1):
        ours, truth = unit_m.draw_scenario(unit_m.SCENARIO, unit_m.PRIMARY_N, replicate)
        theirs, expected = canonical_multi_arm.draw_scenario(
            unit_m.SCENARIO, unit_m.PRIMARY_N, replicate
        )
        pd.testing.assert_frame_equal(ours, theirs)
        assert truth == expected
    beyond = unit_m.REGISTERED_REPLICATES
    assert replicate_seed(unit_m.STUDY, unit_m.SCENARIO, beyond) == replicate_seed(
        canonical_multi_arm.STUDY, unit_m.SCENARIO, beyond
    )


def test_unit_m_cell_reads_the_registered_outcome_correct_draws_and_settings() -> None:
    registered_cell = unit_m.registered_cell()
    cell = unit_m.property_cell()
    assert (registered_cell.property, registered_cell.cell) == (
        "double_robustness",
        "outcome_correct",
    )
    assert (cell.property, cell.cell, cell.role) == (
        "double_robustness",
        "outcome_correct",
        "positive",
    )
    assert cell.n == registered_cell.n == 1_000
    assert cell.seed == registered_cell.seed
    assert cell.estimand == registered_cell.estimand
    assert cell.dgp == registered_cell.dgp
    assert cell.replicates == unit_m.BUDGETS["outcome_correct"] == 800
    assert registered_cell.replicates == unit_m.FLOORS["outcome_correct"] == 600
    assert unit_m.property_cell(2).replicates == 2
    ours = unit_m.cell_settings(cell)
    theirs = registered_multi_arm._settings(registered_cell)
    assert theirs["reduced_crossfit"] == "pooled"
    assert ours["reduced_crossfit"] == "foldwise"
    assert set(ours) == set(theirs)
    for key in set(ours) - {"outcome_learner", "treatment_learner", "reduced_crossfit"}:
        assert repr(ours[key]) == repr(theirs[key]), key
    for key in ("outcome_learner", "treatment_learner"):
        assert repr(ours[key]) == repr(theirs[key]), key
    # The registered cell's sample seeds are the head of this cell's stream.
    budgets: dict[int, int] = {}
    for other in registered_multi_arm.cells():
        budgets[other.seed] = max(budgets.get(other.seed, 0), other.replicates)
    registered_seeds = sample_seed_streams(budgets)[registered_cell.seed]
    own_seeds = sample_seed_streams({cell.seed: cell.replicates + 200})[cell.seed]
    assert own_seeds[: registered_cell.replicates] == registered_seeds
    (declared,) = unit_m.declared_cells()
    assert (declared.property, declared.cell, declared.seed, declared.replicates) == (
        cell.property,
        cell.cell,
        cell.seed,
        cell.replicates,
    )


def test_unit_m_rule_sizes_the_primary_rows_and_the_cell_jointly() -> None:
    rule = unit_m.rule()
    assert unit_m.PLANNING_GROUPS == {
        "primary": "m_primary",
        "outcome_correct": "m_outcome_correct",
    }
    assert rule.pilots["primary"].rows == tuple(sorted(unit_m.ESTIMANDS))
    assert rule.pilots["primary"].calibrated
    assert rule.pilots["outcome_correct"].rows == ("outcome_correct",)
    assert not rule.pilots["outcome_correct"].calibrated
    assert rule.units == {
        "primary": (("primary",), 1_000),
        "outcome_correct": (("outcome_correct",), 600),
    }
    assert rule.unit_target == pytest.approx(0.95**0.5)
    assert rule.budget("primary") == (1_200, pytest.approx(0.9904, abs=PASS_TOLERANCE), True)
    assert rule.budget("outcome_correct") == (
        800,
        pytest.approx(0.99655, abs=PASS_TOLERANCE),
        True,
    )
    assert {unit: rule.budget(unit)[0] for unit in rule.units} == dict(unit_m.BUDGETS)
    assert rule.family_pass(unit_m.BUDGETS) >= 0.95
    assert unit_m.FLOORS == {"primary": 1_000, "outcome_correct": 600}


def test_the_pilot_fits_and_sizes_the_multi_arm_outcome_correct_cell() -> None:
    assert pilot.GROUPS["m_outcome_correct"] == (unit_m, 1_000)
    rows = pd.DataFrame(
        [row for replicate in range(3) for row in pilot.fit("m_outcome_correct", replicate)]
    )
    assert set(rows["estimand"]) == {"outcome_correct"}
    assert rows["seconds"].gt(0).all()
    summary = pilot.summarize(rows)
    assert list(summary["group"]) == ["m_outcome_correct"]
    assert pilot.sizing(rows)[0].startswith("unit M: ")
    assert pilot.seed("m_outcome_correct", 0) != unit_m.registered_cell().seed


# ------------------------------------------------------------------------- unit S, primary
def test_unit_s_primary_adds_the_marginal_ate_to_the_canonical_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = unit_s.STUDY
    assert record.scenarios == {
        canonical_stratified.SCENARIO: (*canonical_stratified.ESTIMANDS, "ate")
    }
    assert "ate" not in canonical_stratified.ESTIMANDS
    assert record.n == canonical_stratified.PRIMARY_N == 2_000
    assert record.property_cells == unit_s.PROPERTY_CELLS
    truths = stratified_alternating_law.paper_truths()
    mixture = sum(
        mass * truths[f"ate[V={stratum}]"]
        for stratum, mass in zip(
            stratified_alternating_law.STRATA, stratified_alternating_law.PAPER_P_V, strict=True
        )
    )
    assert truths["ate"] == pytest.approx(mixture, abs=1e-12)
    calls: list[tuple[str, str]] = []

    def spy(frame: pd.DataFrame, configuration: str, *, reduced_crossfit: str) -> str:
        calls.append((configuration, reduced_crossfit))
        return "fitted"

    monkeypatch.setattr(canonical_stratified, "fit_cleverly", spy)
    assert unit_s.fit_cleverly(pd.DataFrame()) == "fitted"
    assert calls == [("both_correct", "foldwise")]


def test_unit_s_primary_draws_are_the_registered_stratified_draws() -> None:
    for replicate in (0, 1):
        ours, truth = unit_s.draw_scenario(unit_s.SCENARIO, unit_s.PRIMARY_N, replicate)
        theirs, expected = canonical_stratified.draw_scenario(
            unit_s.SCENARIO, unit_s.PRIMARY_N, replicate
        )
        pd.testing.assert_frame_equal(ours, theirs)
        assert truth == expected
    beyond = unit_s.REGISTERED_REPLICATES
    assert replicate_seed(unit_s.STUDY, unit_s.SCENARIO, beyond) == replicate_seed(
        canonical_stratified.STUDY, unit_s.SCENARIO, beyond
    )


def test_unit_s_primary_rule_sizes_every_row_but_the_expected_finite_sample_row() -> None:
    """``ey[1][V=2]``'s pooled coverage reads 0.90, so no budget makes it pass.

    It is declared before the run as an expected finite-sample row and left out of the sizing
    family; with it in, the pooled rows size nothing up to the ceiling.
    """
    assert unit_s.EXPECTED_FINITE_SAMPLE == ("ey[1][V=2]",)
    assert set(unit_s.SIZED_ESTIMANDS) == set(unit_s.ESTIMANDS) - {"ey[1][V=2]"}
    assert unit_s.PLANNING_GROUP == "s_primary"
    piloted = committed_planning()[0]["s_primary"]
    assert isinstance(piloted, Pilot)
    assert set(piloted.rows) == set(unit_s.ESTIMANDS)
    rule = unit_s.rule()
    assert set(rule.pilots["primary"].rows) == set(unit_s.SIZED_ESTIMANDS)
    assert rule.units == {"primary": (("primary",), unit_s.PRIMARY_FLOOR)}
    assert rule.budget("primary") == (4_000, pytest.approx(0.95725, abs=PASS_TOLERANCE), True)
    assert rule.budget("primary")[0] == unit_s.PRIMARY_REPLICATES
    # The diagnosis item is closed: the both-correct marginal ATE reads well below 0.12 SD.
    assert abs(piloted.bias[piloted.rows.index("ate")]) < 0.05
    # The rule drops the row from a pilot that carries it; with it kept, nothing sizes.
    rows = pd.read_parquet(canonical_stratified.STUDY.artifact("replicates.parquet"))
    own = rows.loc[rows["implementation"] == canonical_stratified.STUDY.implementation]
    full = pilot_from_rows(own, source="test", estimands=canonical_stratified.ESTIMANDS)
    assert full.coverage[full.rows.index("ey[1][V=2]")] == pytest.approx(0.90)
    assert "ey[1][V=2]" not in unit_s.rule({"s_primary": full}).pilots["primary"].rows
    kept = Rule(pilots={"primary": full}, units={"primary": (("primary",), 1_000)}, seed=1)
    assert kept.budget("primary")[2] is False


def test_the_pooled_small_stratum_coverage_is_the_committed_one() -> None:
    performance = pd.read_csv(canonical_stratified.STUDY.artifact("performance-tests.csv"))
    (coverage,) = performance.loc[
        (performance["implementation"] == canonical_stratified.STUDY.implementation)
        & (performance["estimand"] == "ey[1][V=2]"),
        "coverage",
    ]
    assert coverage == pytest.approx(unit_s.POOLED_SMALL_STRATUM_COVERAGE)


SMALL_STRATUM_RED: dict[str, Any] = {
    "passed": False,
    "bias_equivalent": True,
    "coverage_ci_lower": 0.87,
    "coverage_ci_upper": 0.93,
    "se_ratio_ci_lower": 0.88,
    "se_ratio_ci_upper": 1.01,
}


@pytest.mark.parametrize(
    ("change", "reading"),
    [
        ({"passed": True}, unit_s.SMALL_STRATUM_CLEAN),
        ({}, unit_s.SMALL_STRATUM_FINITE_SAMPLE),
        ({"bias_equivalent": False}, unit_s.SMALL_STRATUM_DEFECT),
        ({"se_ratio_ci_upper": 0.99}, unit_s.SMALL_STRATUM_DEFECT),
        ({"coverage_ci_lower": 0.91}, unit_s.SMALL_STRATUM_DEFECT),
        ({"coverage_ci_upper": 0.89}, unit_s.SMALL_STRATUM_DEFECT),
    ],
)
def test_the_small_stratum_reading_follows_its_declaration(
    change: dict[str, Any], reading: str
) -> None:
    assert unit_s.small_stratum_reading({**SMALL_STRATUM_RED, **change}) == reading


# --------------------------------------------------------------------------- unit S, ladder
def test_the_ladder_is_the_stratified_records_property_cells() -> None:
    assert unit_s.STUDY.properties_module == ladder.__name__
    cells = unit_s.STUDY.property_cells[ladder.FAMILY]
    assert len(cells) == len(ladder.LABELS) * len(ladder.REGIMES) * (len(ladder.SIZES) + 1) == 32
    assert ladder.SIZES == (2_000, 4_000, 8_000)
    assert ladder.READ_LABELS == ("marginal_ate", "v1_ate")
    produced = {
        name.split("/", 1)[1] for key in ladder.payloads(2) for name in ladder._cells_of(key)
    }
    rates = {
        ladder.rate_cell(label, regime) for label in ladder.LABELS for regime in ladder.REGIMES
    }
    assert produced | rates == set(cells)
    assert ladder.PROPERTY_RULES[0].families == frozenset({ladder.FAMILY})
    assert not hasattr(ladder, "STUDY")
    assert not (
        ROOT / "tests" / "canonical" / "stratified_drtmle_foldwise_ladder" / "regenerate.py"
    ).exists()


def test_the_ladder_budgets_are_sized_by_the_predictive_rule() -> None:
    budgets = ladder.BUDGETS
    assert budgets[ladder.POSITIVE] == (6_000, 800, 6_000)
    assert budgets[ladder.CONTROL] == (400, 400, 400)
    assert ladder.VERDICT_REPLICATES == ladder.REGISTERED_REPLICATES == budgets[ladder.POSITIVE][1]
    assert ladder.RESOLUTION_TARGET == 0.25
    assert (ladder.RESOLUTION_PROBABILITY, ladder.CONTROL_PROBABILITY) == (0.95, 0.99)
    assert ladder.PLANNING_GROUPS == ("ladder_tc_8000", "ladder_tc_2000")
    plan = ladder.positive_plan()
    assert plan.labels == ladder.READ_LABELS == ("marginal_ate", "v1_ate")
    assert plan.size == 8_000 and plan.replicates == 3_000
    assert plan.standardized == pytest.approx((0.2461, 0.1454), abs=1e-4)
    # The second-order planning values size both slopes at 6,000 per outer rung.
    assert ladder.planned_outer_budget() == (
        6_000,
        pytest.approx(0.962, abs=PASS_TOLERANCE),
        True,
    )
    assert ladder.planned_outer_budget()[0] == budgets[ladder.POSITIVE][0]
    # Planned from the n = 8,000 rung alone, the ladder would not size by the ceiling.
    top = committed_planning()[0]["ladder_tc_8000"]
    assert isinstance(top, LadderPlan)
    alone = predictive_outer_budget(
        top,
        ladder.SIZES,
        ladder.VERDICT_REPLICATES,
        target=ladder.RESOLUTION_TARGET,
        level=0.99,
        probability=ladder.RESOLUTION_PROBABILITY,
        floor=ladder.VERDICT_REPLICATES,
        seed=ladder.PLANNING_SEED,
    )
    assert alone == (8_000, pytest.approx(0.894, abs=PASS_TOLERANCE), False)
    # The control's first-order bias is large, so 400 per rung fails to contract surely.
    control = ladder.control_plan()
    assert control.labels == ladder.READ_LABELS and control.size == 2_000
    assert control.replicates == 800
    assert ladder.planned_control_budget() == (400, pytest.approx(1.0), True)
    assert ladder.planned_control_budget()[0] == budgets[ladder.CONTROL][0]
    # The middle rung's weight is zero, so its budget moves nothing.
    z = (0.3, 0.3 / np.sqrt(2), 0.15)
    at = slope_half_width(z, (6_000, 800, 6_000), ladder.SIZES, 0.99)
    assert slope_half_width(z, (6_000, 50, 6_000), ladder.SIZES, 0.99) == pytest.approx(at)


def test_the_ladder_declares_the_x8_acceptance_verbatim() -> None:
    assert ladder.X8_ACCEPTANCE == X8_ACCEPTANCE


def test_the_ladder_paired_rung_reads_the_x8_stream() -> None:
    assert ladder.PAIRED_RUNG == ("treatment_correct", 2_000)
    assert x8.ROBUSTNESS_STREAMS["treatment_correct"] == "treatment_correct"
    assert x8.BUDGETS["treatment_correct"] == ladder.REGISTERED_REPLICATES
    for replicate in (0, 1):
        ours, truth = ladder._draw_rung((("treatment_correct", 2_000, replicate, 2),))
        seed = x8._seed("treatment_correct", replicate)
        theirs, expected = x8._draw(
            (("double_robustness", "treatment_correct", replicate, 2_000, 800, seed),)
        )
        pd.testing.assert_frame_equal(ours, theirs)
        assert truth == expected
    beyond = ladder.REGISTERED_REPLICATES
    assert ladder.sample_seed("treatment_correct", 2_000, beyond) == x8._seed(
        "treatment_correct", beyond
    )


def test_every_other_ladder_rung_draws_a_fresh_stream() -> None:
    fresh = [
        (regime, size)
        for regime in ladder.REGIMES
        for size in ladder.SIZES
        if (regime, size) != ladder.PAIRED_RUNG
    ]
    seeds = {
        rung: {ladder.sample_seed(*rung, replicate) for replicate in range(50)} for rung in fresh
    }
    union = set().union(*seeds.values())
    assert len(union) == 50 * len(fresh)
    registered_streams = {
        x8._seed(stream, replicate)
        for stream in (*x8.ROBUSTNESS_STREAMS.values(), x8.CALIBRATION_STREAM)
        for replicate in range(50)
    }
    paired = {ladder.sample_seed(*ladder.PAIRED_RUNG, replicate) for replicate in range(50)}
    assert not union & registered_streams
    assert not union & paired
    for rung in fresh:
        assert ladder.sample_seed(*rung, 0) == stream_seed(
            unit_s.STUDY, "property_sample", f"{rung[0]}_n{rung[1]}", 0
        )


def test_the_ladder_fits_the_canonical_stratified_fit_with_foldwise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def spy(frame: pd.DataFrame, configuration: str, *, reduced_crossfit: str) -> object:
        calls.append((configuration, reduced_crossfit))
        raise RuntimeError("spied")

    monkeypatch.setattr(canonical_stratified, "fit_cleverly", spy)
    for regime in ladder.REGIMES:
        with pytest.raises(RuntimeError, match="spied"):
            ladder._fit_key(((regime, 2_000, 0, 2),), (pd.DataFrame(), {}))
    assert calls == [("treatment_correct", "foldwise"), ("both_wrong", "foldwise")]


def test_the_ladder_reads_level_rows_at_the_verdict_budget() -> None:
    rows = pd.DataFrame(
        {
            "property": ladder.FAMILY,
            "cell": ["marginal_ate__treatment_correct_n2000"] * 5
            + ["marginal_ate__both_wrong_n2000"] * 3,
            "role": ["positive"] * 5 + ["control"] * 3,
            "replicate": [0, 1, 2, 3, 4, 0, 1, 2],
            "requested_replicates": [5] * 5 + [3] * 3,
            "failed_replicates": 0,
        }
    )
    kept = ladder.at_verdict_budget(rows, verdict_replicates=3)
    positive = kept.loc[kept["role"] == "positive"]
    assert sorted(positive["replicate"]) == [0, 1, 2]
    assert set(positive["requested_replicates"]) == {3}
    assert len(kept.loc[kept["role"] == "control"]) == 3
    with pytest.raises(ValueError, match="fewer than"):
        ladder.at_verdict_budget(rows, verdict_replicates=6)
    assert np.isfinite(ladder.VERDICT_REPLICATES)
