"""Design of the four Fix-6 records of the fold-wise DR-TMLE default.

``drtmle-foldwise-n1500`` (unit B), ``multi-arm-drtmle-foldwise`` (unit M),
``stratified-drtmle-foldwise`` (unit S, primary) and ``stratified-drtmle-foldwise-ladder``
(unit S, ladder).  Each test names the part of the declaration it pins: the record, the
pairing of its draws with the registered study they re-read, the fresh streams, the planning
budgets and their sizing rule, the guard that refuses a declared run while they are planning
values, and, for unit B, the nonzero witness that its fold-wise path is not wired pooled.  No
test runs a study; the witness fits two samples.
"""

from __future__ import annotations

import inspect
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from tests.studies import canonical_drtmle, drtmle_properties, stratified_alternating_law
from tests.studies import canonical_multi_arm_drtmle as canonical_multi_arm
from tests.studies import canonical_stratified_drtmle as canonical_stratified
from tests.studies import drtmle_foldwise as foldwise
from tests.studies import drtmle_foldwise_n1500 as unit_b
from tests.studies import multi_arm_drtmle_foldwise as unit_m
from tests.studies import stratified_drtmle_foldwise as unit_s
from tests.studies import stratified_drtmle_foldwise_ladder as ladder
from tests.studies import stratified_drtmle_properties as x8
from tests.studies.drtmle_foldwise_sizing import slope_half_width
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.registry import ROOT, StudyRecord, registered
from tests.studies.evidence.seeds import replicate_seed, stream_seed

MODULES: dict[str, ModuleType] = {
    "drtmle-foldwise-n1500": unit_b,
    "multi-arm-drtmle-foldwise": unit_m,
    "stratified-drtmle-foldwise": unit_s,
    "stratified-drtmle-foldwise-ladder": ladder,
}
POLICIES = {
    "drtmle-foldwise-n1500": "gated",
    "multi-arm-drtmle-foldwise": "gated",
    "stratified-drtmle-foldwise": "gated",
    "stratified-drtmle-foldwise-ladder": "reporting",
}
DIRECTORIES = {
    "drtmle-foldwise-n1500": "drtmle_foldwise_n1500",
    "multi-arm-drtmle-foldwise": "multi_arm_drtmle_foldwise",
    "stratified-drtmle-foldwise": "stratified_drtmle_foldwise",
    "stratified-drtmle-foldwise-ladder": "stratified_drtmle_foldwise_ladder",
}

#: The smallest |fold-wise - pooled| difference the unit B witness accepts on its draw.  The
#: draw reads 0.026; float drift between two pooled fits of one draw is about 1e-16.
FOLDWISE_WITNESS_FLOOR = 1e-3
#: How closely the pooled refit reproduces the committed RM18 estimate.
POOLED_REPRODUCTION_TOLERANCE = 1e-10


def _record(slug: str) -> StudyRecord:
    (record,) = (study for study in registered() if study.slug == slug)
    return record


# ------------------------------------------------------------------------------ every record
@pytest.mark.parametrize("slug", sorted(MODULES))
def test_the_record_registers_with_its_declared_policy_and_no_comparator(slug: str) -> None:
    module = MODULES[slug]
    record = _record(slug)
    assert record is module.STUDY
    assert record.reference is None
    assert record.pending_reference is None
    assert record.comparator_state == "none"
    assert record.publication_policy == POLICIES[slug]
    assert record.implementation == foldwise.IMPLEMENTATION
    assert dict(record.failure_bound) == {foldwise.IMPLEMENTATION: 0.01}
    assert module.FAILURE_BOUND == 0.01
    assert module.PROPERTY_WITNESS is property_verdicts.SAMPLE_WITNESS
    assert record.artifacts == ROOT / "tests" / "canonical" / DIRECTORIES[slug]
    assert (record.artifacts / "__init__.py").is_file()
    assert (record.artifacts / "regenerate.py").is_file()
    assert record.runner() is module
    assert record.properties() is module
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
    # The two records with their own primary stream draw a seed no other study uses.
    for slug in ("drtmle-foldwise-n1500", "stratified-drtmle-foldwise-ladder"):
        assert ours[slug].seed not in taken
    # The two paired primaries share their canonical study's seed on purpose.
    assert unit_m.SEED == canonical_multi_arm.SEED
    assert unit_s.SEED == canonical_stratified.SEED


@pytest.mark.parametrize("slug", sorted(MODULES))
def test_a_declared_run_is_refused_while_the_budgets_are_planning_values(
    slug: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = MODULES[slug]
    assert module.BUDGETS_DECLARED is False
    with pytest.raises(RuntimeError, match=r"planning|placeholder"):
        module.assert_declared()
    # The guard runs before any draw, at the declared primary budget.
    with pytest.raises(RuntimeError, match=r"planning|placeholder"):
        module.draw_and_fit(replicates=module.PRIMARY_REPLICATES, n=40, n_jobs=1)
    if hasattr(module, "PRIMARY_DESIGN_OPEN"):
        assert module.PRIMARY_DESIGN_OPEN is True
        monkeypatch.setattr(module, "PRIMARY_DESIGN_OPEN", False)
        with pytest.raises(RuntimeError, match="planning"):
            module.assert_declared()
        with pytest.raises(RuntimeError, match="planning"):
            module.generate_property_rows(n_jobs=1)
    monkeypatch.setattr(module, "BUDGETS_DECLARED", True)
    module.assert_declared()


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
    assert unit_b.PLANNING_BUDGETS["cells"] == unit_b.REGISTERED_REPLICATES
    assert unit_b.PLANNING_BUDGETS["primary"] == unit_b.PRIMARY_REPLICATES == 1_000
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


def test_unit_b_rule_records_its_budget() -> None:
    rule = unit_b.RULE
    assert {name: pilot.bias for name, pilot in rule.pilots.items()} == {
        "outcome_correct": (0.08,),
        "treatment_correct": (0.12,),
    }
    assert all(pilot.replicates == 800 for pilot in rule.pilots.values())
    assert all(not pilot.se_ratio for pilot in rule.pilots.values())
    assert rule.units == {"cells": (("outcome_correct", "treatment_correct"), 800)}
    assert rule.unit_target == pytest.approx(0.95)
    replicates, value, sized = rule.budget("cells")
    # The planning values of plan section 5 size the cells at 2,000, above the paired floor.
    assert (replicates, sized) == (2_000, True)
    assert value == pytest.approx(0.955, abs=0.003)
    assert rule.unit_pass("cells", 1_600) < rule.unit_target


def test_unit_b_placeholder_primary_gates_nothing() -> None:
    record = unit_b.STUDY
    assert record.scenarios == dict.fromkeys(unit_b.SCENARIOS, canonical_drtmle.ESTIMANDS)
    assert record.reported_scenarios == frozenset(unit_b.SCENARIOS)
    assert record.n == unit_b.PRIMARY_N
    for scenario in unit_b.SCENARIOS:
        assert replicate_seed(record, scenario, 0) != replicate_seed(
            canonical_drtmle.STUDY, scenario, 0
        )


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
    assert record.property_cells == {}
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


def test_unit_m_rule_on_the_registered_pooled_rows() -> None:
    pilot = unit_m.registered_pilot()
    assert pilot.rows == unit_m.ESTIMANDS
    assert pilot.replicates == canonical_multi_arm.PRIMARY_REPLICATES
    assert pilot.bias[:3] == (0.046, 0.0332, 0.0168)
    assert pilot.se_ratio[:3] == (0.999, 1.0413, 1.0366)
    rule = unit_m.rule()
    assert next(iter(rule.grid("primary"))) == 1_200
    replicates, value, sized = rule.budget("primary")
    assert (replicates, sized) == (1_200, True)
    assert value == pytest.approx(0.960, abs=0.003)
    assert unit_m.PLANNING_REPLICATES == unit_m.PRIMARY_FLOOR == 1_000


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
    assert record.property_cells == {}
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


def test_unit_s_primary_rule_on_the_registered_pooled_rows() -> None:
    """The pooled rows do not size the unit: ``ey[1][V=2]``'s pooled coverage reads 0.90.

    That is ``X8-drtmle-small-stratum``'s finite-sample red, carried here as a planning value.
    The rule refuses to size it, so the sizing pilot's fold-wise rows decide the budget.
    """
    pilots = unit_s.registered_pilots()
    assert pilots["primary"].rows == canonical_stratified.ESTIMANDS
    assert pilots["marginal"].rows == ("ate",)
    assert pilots["marginal"].replicates == 2_000
    index = pilots["primary"].rows.index("ey[1][V=2]")
    assert pilots["primary"].coverage[index] == pytest.approx(0.90)
    rule = unit_s.rule()
    assert rule.budget("primary")[2] is False
    assert rule.unit_pass("primary", 2_400, ("bias", "se_ratio")) > 0.9
    assert rule.unit_pass("primary", 8_000, ("coverage",)) < 0.5


# --------------------------------------------------------------------------- unit S, ladder
def test_the_ladder_declares_every_level_and_rate_cell() -> None:
    cells = ladder.STUDY.property_cells[ladder.FAMILY]
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
    assert ladder.STUDY.reported_scenarios == frozenset({ladder.PRIMARY_SCENARIO})


def test_the_ladder_budgets_reach_the_resolution_target() -> None:
    budgets = ladder.PLANNING_BUDGETS
    assert budgets[ladder.POSITIVE] == (3_300, 800, 3_300)
    assert budgets[ladder.CONTROL] == (400, 400, 400)
    assert ladder.VERDICT_REPLICATES == ladder.REGISTERED_REPLICATES == budgets[ladder.POSITIVE][1]
    assert ladder.RESOLUTION_TARGET == 0.25
    assert ladder.planned_outer_budget() == budgets[ladder.POSITIVE][0]
    marginal = ladder.PLANNING_STANDARDIZED_BIAS["marginal_ate"]
    assert marginal[0] == 0.2905
    assert marginal[2] == pytest.approx(marginal[0] / 2)
    at = slope_half_width(marginal, (3_300, 800, 3_300), ladder.SIZES, 0.99)
    below = slope_half_width(marginal, (3_200, 800, 3_200), ladder.SIZES, 0.99)
    assert below > ladder.RESOLUTION_TARGET >= at
    # The middle rung's weight is zero, so its budget moves nothing.
    assert slope_half_width(marginal, (3_300, 50, 3_300), ladder.SIZES, 0.99) == pytest.approx(at)


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
            ladder.STUDY, "property_sample", f"{rung[0]}_n{rung[1]}", 0
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
