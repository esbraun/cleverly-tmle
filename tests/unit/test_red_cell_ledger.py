"""The red-cell ledger, against the committed verdicts, the roadmap and the registry.

``tests/studies/evidence/red_cells.py`` says why the ledger is generated.  These tests hold the
four relations it exists for: every red row has an owner, every owner has a red row, every owner
is an ask the roadmap lists, and every red row sits in a ``reporting`` study.  The published
page must be what the committed tables render.

Several relations here are empty on the committed data, and an empty check passes whether or
not it can see anything.  Each one therefore has a deliberate mutation beside it that must fail.
"""

from __future__ import annotations

import dataclasses
import math
from importlib import import_module
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tests.studies.evidence import failures, property_failures, red_cells
from tests.studies.evidence.comparison import empty_equivalence
from tests.studies.evidence.red_cells import (
    ACCEPTED_BY_EXCEPTION,
    BLOCKS,
    EXCLUDED_HOOKS,
    FRAMES,
    OWNERS,
    VERDICT_COLUMNS,
    RedKey,
    RedRow,
    check_verdict_columns,
    findings,
    frames,
    owner_ids,
    owners,
    paired_legs,
    published,
    red_rows,
    rendered,
)
from tests.studies.evidence.registry import StudyRecord, registered

STUDIES = registered()
BY_SLUG = {study.slug: study for study in STUDIES}
POLICIES = {study.slug: study.publication_policy for study in STUDIES}

#: The one committed paired row that is red for its calibration leg alone.
SHIFT = "shift-policies"
SHIFT_RED = ("continuous_modified_policy", "ate_policy[+0.25 vs natural course]")


@pytest.fixture(scope="module")
def tables() -> dict[str, dict[str, pd.DataFrame]]:
    return {study.slug: frames(study) for study in STUDIES}


@pytest.fixture(scope="module")
def rows(tables: dict[str, dict[str, pd.DataFrame]]) -> list[RedRow]:
    return [row for study in STUDIES for row in red_rows(study, tables[study.slug])]


# --- the four relations, on the committed data ---------------------------------------------


def test_every_red_row_has_an_owner(rows: list[RedRow]) -> None:
    unowned = sorted({row.key for row in rows} - set(OWNERS) - set(ACCEPTED_BY_EXCEPTION))
    assert unowned == [], (
        f"these committed verdicts are red and no roadmap ask owns them: {unowned}. Add each "
        f"to red_cells.CLAIMS under the ask that will close it"
    )


def test_every_owner_entry_is_still_red(rows: list[RedRow]) -> None:
    """The other direction of the ratchet.  A cell that turns green must leave the ledger."""
    stale = sorted(set(OWNERS) - {row.key for row in rows})
    assert stale == [], (
        f"red_cells.CLAIMS owns these verdicts, but none of them is red: {stale}. Remove them"
    )


def test_a_red_row_needs_the_reporting_policy(rows: list[RedRow]) -> None:
    gated = sorted(
        {
            row.key.slug
            for row in rows
            if POLICIES[row.key.slug] != "reporting" and row.key not in ACCEPTED_BY_EXCEPTION
        }
    )
    assert gated == [], f"these studies publish a red verdict but are not `reporting`: {gated}"


def test_the_ledger_has_no_findings(rows: list[RedRow]) -> None:
    assert findings(rows, OWNERS, POLICIES, owner_ids(), ACCEPTED_BY_EXCEPTION) == []


# --- the named exceptions: gated studies' admitted red rows --------------------------------------

ACCEPTED_KEY = RedKey(
    "adaptive-tmle-selective", "truth", "cleverly-adaptive-tmle-selective/s6_cv/psi"
)
#: Admitted by exception but not accepted as valid: owned until Fix-3 re-reads it.
OWNED_EXCEPTION_KEY = RedKey(
    "stratified-drtmle-foldwise",
    "truth",
    "cleverly-drtmle-foldwise/stratified_both_correct/ey[1][V=2]",
)
#: Each exception and the dated written decision of the project owner that granted it.
EXCEPTION_DECISIONS = {ACCEPTED_KEY: "2026-11-10", OWNED_EXCEPTION_KEY: "2026-11-13"}


def test_the_exceptions_admit_exactly_the_two_decided_rows() -> None:
    assert set(ACCEPTED_BY_EXCEPTION) == set(EXCEPTION_DECISIONS)
    for key, date in EXCEPTION_DECISIONS.items():
        entry = ACCEPTED_BY_EXCEPTION[key]
        assert date in entry.reason and "Eric" in entry.reason


def test_the_selective_row_is_unowned_and_the_stratified_row_keeps_its_owner() -> None:
    assert ACCEPTED_KEY not in OWNERS
    assert OWNERS[OWNED_EXCEPTION_KEY] == "X8-drtmle-small-stratum"


@pytest.mark.parametrize("key", sorted(EXCEPTION_DECISIONS), ids=lambda key: key.slug)
def test_the_exception_row_is_gated_and_still_red(rows: list[RedRow], key: RedKey) -> None:
    """Ratchet 1: the cell turns green, or the study turns `reporting`, and the entry goes."""
    assert key in {row.key for row in rows}, "the cell is green; remove the exception"
    assert POLICIES[key.slug] == "gated", "the study is reporting; remove the exception"


@pytest.mark.parametrize("key", sorted(EXCEPTION_DECISIONS), ids=lambda key: key.slug)
def test_the_exception_covers_only_the_accepted_run(key: RedKey) -> None:
    """Ratchet 2: a rerun writes another replicates.parquet, and the entry fails until removed."""
    import json

    manifest = json.loads(BY_SLUG[key.slug].artifact("manifest.json").read_text(encoding="utf-8"))
    recorded = manifest["sha256"]["replicates.parquet"]
    assert recorded == ACCEPTED_BY_EXCEPTION[key].replicates_sha256, (
        "the study was rerun; the exception covered the run it names. Remove it"
    )


def test_an_owned_exception_is_still_held_to_its_owner(rows: list[RedRow]) -> None:
    """The owned exception lifts the policy only: without its owner it is an unowned red row."""
    row = next(r for r in rows if r.key == OWNED_EXCEPTION_KEY)
    owned = {OWNED_EXCEPTION_KEY: "X8-drtmle-small-stratum"}
    ids = ["X8-drtmle-small-stratum"]
    accepted = [OWNED_EXCEPTION_KEY]
    assert findings([row], owned, POLICIES, ids, accepted) == []
    assert findings([row], owned, POLICIES, ids) != []
    assert findings([], owned, POLICIES, ids, accepted) == [
        f"{OWNED_EXCEPTION_KEY} has an owner but is not red",
        f"{OWNED_EXCEPTION_KEY} is accepted by exception but is not red",
    ]


def test_the_exception_is_keyed_to_one_row_and_not_its_study(rows: list[RedRow]) -> None:
    """A second red row of the same study, or another gated study's red row, is still refused."""
    other = dataclasses.replace(next(r for r in rows if r.key == ACCEPTED_KEY))
    sibling = dataclasses.replace(
        other, key=dataclasses.replace(ACCEPTED_KEY, key="x/s6_pooled/psi")
    )
    problems = findings([other, sibling], {}, POLICIES, [], ACCEPTED_BY_EXCEPTION)
    assert any("s6_pooled" in problem and "no owner" in problem for problem in problems)
    assert not any("s6_cv" in problem for problem in problems)


def test_a_green_exception_is_a_finding(rows: list[RedRow]) -> None:
    problems = findings([], {}, POLICIES, [], ACCEPTED_BY_EXCEPTION)
    assert any("accepted by exception but is not red" in problem for problem in problems)


def test_every_owner_is_an_id_the_roadmap_lists() -> None:
    ids = owner_ids()
    assert len(ids) == len(set(ids)), f"the roadmap's id column repeats a name: {ids}"
    unknown = sorted(set(OWNERS.values()) - set(ids))
    assert unknown == [], f"these owners are not in the roadmap's id column: {unknown}"


def test_every_committed_paired_verdict_follows_its_legs(
    tables: dict[str, dict[str, pd.DataFrame]],
) -> None:
    """Every paired row of every study, not only the red ones.

    A green row whose committed ``passed`` disagreed with its legs would hide a red cell from
    the ledger, so the agreement has to hold on both sides of the verdict.
    """
    disagreements = []
    for study in STUDIES:
        performance = tables[study.slug]["performance"]
        valid = performance.set_index(["implementation", "scenario", "estimand"])["passed"]
        # A cell that lost a replicate carries its failure-accounting gate, which can withdraw
        # the conclusion its legs give (``failures.accounted_conclusion``).
        gates = failures.cell_gates(tables[study.slug]["accounting"], "equivalence")
        for row in tables[study.slug]["equivalence"].itertuples():
            subject_valid = bool(valid.loc[(study.implementation, row.scenario, row.estimand)])
            legs = paired_legs(row, study.margins, subject_valid)
            rebuilt = (legs.passed, legs.conclusion)
            gate = gates.get((str(row.scenario), str(row.estimand)))
            if gate is not None:
                conclusion, passed = failures.accounted_conclusion(
                    legs.conclusion,
                    equivalent=legs.similar and legs.rmse and legs.coverage and legs.calibration,
                    gate=gate,
                )
                rebuilt = (passed, conclusion)
            committed = (bool(row.passed), str(row.comparison_conclusion))
            if rebuilt != committed:
                disagreements.append((study.slug, row.scenario, row.estimand))
    assert disagreements == []


def test_an_accounted_paired_verdict_needs_its_gate(
    tables: dict[str, dict[str, pd.DataFrame]],
) -> None:
    # The witness that the gate branch above is not vacuous: a dose-response-npcausal row
    # whose legs conclude equivalent is committed unaccounted, and only its gate explains it.
    study = BY_SLUG["dose-response-npcausal"]
    frame = tables[study.slug]["equivalence"]
    row = next(row for row in frame.itertuples() if row.estimand == "dose_response[h=2,a=2.5]")
    legs = paired_legs(row, study.margins, True)
    assert (legs.passed, legs.conclusion) == (True, "equivalent")
    gate = failures.cell_gates(tables[study.slug]["accounting"], "equivalence")[
        (str(row.scenario), str(row.estimand))
    ]
    assert gate == (False, failures.WORST_CASE_FAILS)
    assert failures.accounted_conclusion(legs.conclusion, equivalent=True, gate=gate) == (
        failures.UNACCOUNTED,
        False,
    )
    assert (bool(row.passed), str(row.comparison_conclusion)) == (False, failures.UNACCOUNTED)


# --- the published page ---------------------------------------------------------------------


def test_each_block_is_published_once() -> None:
    blocks = published()
    counts = {name: len(blocks.get(name, [])) for name in BLOCKS}
    assert counts == dict.fromkeys(BLOCKS, 1)


@pytest.fixture(scope="module")
def blocks() -> dict[str, list[str]]:
    return rendered()


@pytest.mark.parametrize("name", BLOCKS)
def test_the_published_block_is_the_one_the_results_render(
    blocks: dict[str, list[str]], name: str
) -> None:
    assert published()[name] == [blocks[name]], (
        f"red-cells.md's {name} block is not what the committed tables render. Run "
        f"`python -m tests.studies.evidence.red_cells` rather than editing it by hand"
    )


def test_fill_refuses_a_block_published_twice(tmp_path: Path) -> None:
    page = tmp_path / "red-cells.md"
    text = red_cells.PAGE.read_text(encoding="utf-8")
    page.write_text(text + "\n<!-- generated: red-cells -->\n<!-- /generated -->\n", "utf-8")
    with pytest.raises(LookupError, match="2 times"):
        red_cells.fill(page)


# --- deliberate mutations: the paired verdict ------------------------------------------------


def _shift_tables(
    tables: dict[str, dict[str, pd.DataFrame]],
) -> tuple[StudyRecord, dict[str, pd.DataFrame], pd.Series]:
    study = BY_SLUG[SHIFT]
    copied = {name: frame.copy() for name, frame in tables[SHIFT].items()}
    equivalence = copied["equivalence"]
    red = (equivalence["scenario"] == SHIFT_RED[0]) & (equivalence["estimand"] == SHIFT_RED[1])
    assert red.sum() == 1 and not bool(equivalence.loc[red, "passed"].iloc[0])
    return study, copied, red


def test_the_committed_red_paired_row_fails_on_its_calibration_leg(
    tables: dict[str, dict[str, pd.DataFrame]],
) -> None:
    """A nonzero witness: the one paired row the roadmap once missed is red, and for this leg."""
    study, copied, _ = _shift_tables(tables)
    (row,) = [row for row in red_rows(study, copied) if row.key.kind == "paired"]
    assert row.key.key == "/".join(SHIFT_RED)
    assert row.fails_by == "inconclusive: calibration leg"


@pytest.mark.parametrize("red_row", [True, False], ids=["red to green", "green to red"])
def test_a_flipped_committed_paired_verdict_is_refused(
    tables: dict[str, dict[str, pd.DataFrame]], red_row: bool
) -> None:
    study, copied, red = _shift_tables(tables)
    equivalence = copied["equivalence"]
    target = equivalence.index[red if red_row else ~red][0]
    equivalence.loc[target, "passed"] = not bool(equivalence.loc[target, "passed"])
    with pytest.raises(ValueError, match="but its legs conclude"):
        red_rows(study, copied)


def test_the_legs_decide_the_verdict(tables: dict[str, dict[str, pd.DataFrame]]) -> None:
    """Move the failing leg inside its margin and leave the committed verdict alone.

    The legs now conclude ``equivalent`` while the row still commits ``inconclusive``, so the
    ledger must refuse the row.  A ``paired_legs`` that returned the committed verdict unread
    would accept it, and this test would fail.
    """
    study, copied, red = _shift_tables(tables)
    equivalence = copied["equivalence"]
    equivalence.loc[red, "calibration_excess_upper"] = study.margins.calibration_noninferiority / 2
    with pytest.raises(ValueError, match="but its legs conclude 'equivalent'"):
        red_rows(study, copied)


def test_a_committed_margin_must_match_the_registry(
    tables: dict[str, dict[str, pd.DataFrame]],
) -> None:
    study, copied, red = _shift_tables(tables)
    copied["equivalence"].loc[red, "rmse_noninferiority_margin"] += 1e-6
    with pytest.raises(ValueError, match="committed comparison margins disagree"):
        red_rows(study, copied)


def _one_leg(
    tables: dict[str, dict[str, pd.DataFrame]],
    endpoints: dict[str, float],
    *,
    subject_valid: bool = True,
    committed: tuple[bool, str] = (False, "inconclusive"),
) -> list[RedRow]:
    """The red paired rows of a copy of the shift row, moved so that one leg decides it.

    The calibration leg moves inside its margin first, so every other leg of the row holds and
    its legs would conclude ``equivalent``.  ``endpoints`` then moves one leg, and
    ``committed`` is the verdict that leg should produce.
    """
    study, copied, red = _shift_tables(tables)
    equivalence = copied["equivalence"]
    equivalence.loc[red, "calibration_excess_upper"] = study.margins.calibration_noninferiority / 2
    for column, value in endpoints.items():
        equivalence.loc[red, column] = value
    equivalence.loc[red, "passed"] = committed[0]
    equivalence.loc[red, "comparison_conclusion"] = committed[1]
    if not subject_valid:
        performance = copied["performance"]
        subject = (
            (performance["implementation"] == study.implementation)
            & (performance["scenario"] == SHIFT_RED[0])
            & (performance["estimand"] == SHIFT_RED[1])
        )
        assert subject.sum() == 1
        performance.loc[subject, "passed"] = False
    return [row for row in red_rows(study, copied) if row.key.kind == "paired"]


def _mean_margin(tables: dict[str, dict[str, pd.DataFrame]]) -> float:
    _, copied, red = _shift_tables(tables)
    return float(copied["equivalence"].loc[red, "mean_margin"].iloc[0])


def test_the_rmse_leg_alone_makes_a_row_red(tables: dict[str, dict[str, pd.DataFrame]]) -> None:
    """A nonzero witness for the RMSE leg: a bound just above its margin is red."""
    bound = BY_SLUG[SHIFT].margins.rmse_noninferiority * (1 + 1e-6)
    (row,) = _one_leg(tables, {"rmse_ratio_upper": bound})
    assert row.fails_by == "inconclusive: RMSE leg"


def test_the_similarity_leg_alone_makes_a_row_red(
    tables: dict[str, dict[str, pd.DataFrame]],
) -> None:
    """A nonzero witness for the similarity leg: an upper endpoint just above the margin."""
    (row,) = _one_leg(tables, {"paired_ci_upper": _mean_margin(tables) * (1 + 1e-6)})
    assert row.fails_by == "inconclusive: similarity leg"


def test_an_invalid_subject_cannot_read_superior(
    tables: dict[str, dict[str, pd.DataFrame]],
) -> None:
    """A coverage gain above zero reads ``superior`` only when the subject passes its truth gate.

    The similarity leg fails, so ``superior`` is the only way the row could leave the ledger.
    The control shows that the same endpoints read ``superior`` for a valid subject, so the
    invalid case is red because of the subject's gate and for no other reason.
    """
    endpoints = {
        "paired_ci_upper": _mean_margin(tables) * 1.01,
        "coverage_difference_lower": 0.01,
    }
    (row,) = _one_leg(tables, endpoints, subject_valid=False)
    assert row.fails_by == "inconclusive: similarity leg"
    assert _one_leg(tables, endpoints, committed=(True, "superior")) == []


# --- the verdict columns ---------------------------------------------------------------------


def test_a_missing_verdict_is_refused(tmp_path: Path) -> None:
    """``bool(nan)`` is ``True``, so a blank verdict would otherwise read as a pass."""
    record = BY_SLUG[SHIFT]
    for filename in FRAMES.values():
        pd.read_csv(record.artifact(filename)).to_csv(tmp_path / filename, index=False)
    copy = dataclasses.replace(record, artifacts=tmp_path)
    # Both failure accountings are read too, and are empty for a study where no fit failed.
    assert set(frames(copy)) == {*FRAMES, "accounting", "property_accounting"}
    assert frames(copy)["accounting"].empty
    assert frames(copy)["property_accounting"].empty
    blanked = pd.read_csv(tmp_path / FRAMES["equivalence"])
    blanked["passed"] = blanked["passed"].astype(object)
    blanked.loc[0, "passed"] = None
    blanked.to_csv(tmp_path / FRAMES["equivalence"], index=False)
    with pytest.raises(TypeError, match="'passed' column is object, not bool"):
        frames(copy)


@pytest.mark.parametrize(
    ("table", "column"),
    [(name, column) for name, columns in VERDICT_COLUMNS.items() for column in columns],
)
def test_each_verdict_column_must_be_boolean(
    tables: dict[str, dict[str, pd.DataFrame]], table: str, column: str
) -> None:
    copied = {name: frame.copy() for name, frame in tables[SHIFT].items()}
    copied[table][column] = copied[table][column].astype(float)
    copied[table].loc[copied[table].index[0], column] = float("nan")
    with pytest.raises(TypeError, match=f"'{column}' column is float64"):
        check_verdict_columns(copied)


def test_an_empty_verdict_table_is_allowed() -> None:
    check_verdict_columns(_synthetic())


# --- the hooks the ledger does not read ------------------------------------------------------


def _hook(record: StudyRecord) -> Any:
    return getattr(import_module(record.runner_module), "scientific_failures", None)


def test_every_scientific_failures_hook_is_declared() -> None:
    """A new hook is a new kind of failure, and the ledger page must say it does not read it."""
    hooked = {study.slug for study in STUDIES if _hook(study) is not None}
    assert hooked == set(EXCLUDED_HOOKS)


def _extra(record: StudyRecord) -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(record.artifact(name)) for name in record.extra_artifacts}


def _hook_rows(record: StudyRecord, extra: dict[str, pd.DataFrame]) -> dict[str, int]:
    reported = _hook(record)(extra)
    return {name: len(frame) for name, frame in reported.items() if frame is not None}


GATED_HOOKS = [slug for slug in EXCLUDED_HOOKS if BY_SLUG[slug].publication_policy == "gated"]


def test_some_gated_study_has_a_hook() -> None:
    assert GATED_HOOKS, "the gated-hook checks below would pass on an empty list"


@pytest.mark.parametrize("slug", GATED_HOOKS)
def test_a_gated_study_hook_reports_nothing(slug: str) -> None:
    """The ledger reads no hook, so a gated study's hook must be clean on its committed rows."""
    rows = _hook_rows(BY_SLUG[slug], _extra(BY_SLUG[slug]))
    assert rows and set(rows.values()) == {0}, rows


@pytest.mark.parametrize("slug", GATED_HOOKS)
def test_a_failed_probe_row_is_reported(slug: str) -> None:
    """The witness for the check above: one failed probe row must show up."""
    record = BY_SLUG[slug]
    extra = _extra(record)
    name = next(artifact for artifact, frame in extra.items() if "passed" in frame.columns)
    extra[name].loc[extra[name].index[0], "passed"] = False
    assert sum(_hook_rows(record, extra).values()) == 1


def test_the_page_names_every_excluded_hook() -> None:
    text = red_cells.PAGE.read_text(encoding="utf-8")
    section = text.split("## What counts as red", 1)[1].split("## Who owns each red row", 1)[0]
    missing = [BY_SLUG[slug].name for slug in EXCLUDED_HOOKS if BY_SLUG[slug].name not in section]
    assert missing == []


# --- deliberate mutations: property and truth rows -------------------------------------------


def _synthetic(
    *,
    role: str = "positive",
    passed: bool = True,
    property_passed: bool = True,
    truth_passed: bool = True,
    implementation: str = "cleverly",
) -> dict[str, pd.DataFrame]:
    performance = pd.DataFrame(
        {
            "implementation": [implementation],
            "scenario": ["law"],
            "estimand": ["ate"],
            "passed": [truth_passed],
            "bias_ci_lower": [0.01],
            "bias_ci_upper": [0.02],
            "coverage": [0.9],
            "se_ratio": [1.0],
        }
    )
    properties = pd.DataFrame(
        {
            "property": ["type_i_error"],
            "cell": ["sharp_null"],
            "role": [role],
            "passed": [passed],
            "property_passed": [property_passed],
            "rejection_rate": [0.07],
            "rejection_ci_lower": [0.04],
            "rejection_ci_upper": [0.11],
        }
    )
    return {
        "performance": performance,
        "equivalence": empty_equivalence(),
        "properties": properties,
    }


RECORD = BY_SLUG["canonical-tmle"]


def test_a_family_clause_failure_is_red() -> None:
    (row,) = red_rows(RECORD, _synthetic(passed=True, property_passed=False))
    assert row.key == RedKey(RECORD.slug, "property", "type_i_error/sharp_null")
    assert row.fails_by == "the family's joint clause"


def test_an_own_rule_failure_is_red() -> None:
    (row,) = red_rows(RECORD, _synthetic(passed=False, property_passed=False))
    assert row.fails_by == "its own rule"


def test_a_diagnostic_row_is_never_red() -> None:
    assert (
        red_rows(RECORD, _synthetic(role="diagnostic", passed=False, property_passed=False)) == []
    )


def _property_accounting(reason: str, *, complete_case: float = 0.01) -> pd.DataFrame:
    """The accounting rows of the synthetic ``type_i_error/sharp_null`` cell."""
    base = {
        "property": "type_i_error",
        "cell": "sharp_null",
        "role": "positive",
        "requested": 400,
        "failed": 3,
        "rate": 3 / 400,
        "bound": 0.05,
        "allowed": 20,
        "imputation": "",
        "gate_passed": False,
        "reason": reason,
    }
    return pd.DataFrame(
        [
            {
                **base,
                "leg": "failure bound",
                "complete_case": 3.0,
                "worst_case": 20.0,
                "leg_passed": True,
            },
            {
                **base,
                "leg": "rejection ceiling",
                "complete_case": complete_case,
                "worst_case": -0.02,
                "imputation": "j=3 rejected",
                "leg_passed": False,
            },
        ],
        columns=list(property_failures.PROPERTY_ACCOUNTING_COLUMNS),
    )


def test_a_cell_red_only_by_its_property_accounting_names_it() -> None:
    """A cell green at its complete cases and red under a worst case reads the accounting."""
    tables = _synthetic(passed=False, property_passed=False)
    tables["property_accounting"] = _property_accounting(failures.WORST_CASE_FAILS)
    (row,) = red_rows(RECORD, tables)
    assert row.fails_by == "failure accounting: worst case fails"
    assert row.measured.endswith(", failures 3 of 400 (worst case fails)")
    (finding,) = findings([row], {}, {RECORD.slug: "reporting"}, ["RM18-x"])
    assert "has no owner" in finding


def test_a_cell_red_at_its_complete_case_keeps_its_own_rule() -> None:
    tables = _synthetic(passed=False, property_passed=False)
    tables["property_accounting"] = _property_accounting(
        failures.WORST_CASE_FAILS, complete_case=-0.01
    )
    (row,) = red_rows(RECORD, tables)
    assert row.fails_by == "its own rule"
    assert row.measured.endswith(", failures 3 of 400 (worst case fails)")
    tables["property_accounting"] = _property_accounting(failures.COMPLETE_CASE_FAILS)
    (row,) = red_rows(RECORD, tables)
    assert row.fails_by == "its own rule"


def test_a_not_accountable_cell_red_at_its_complete_case_keeps_its_own_rule() -> None:
    """L5: a not-accountable cell's clause legs give the complete case, and it comes first."""
    tables = _synthetic(passed=False, property_passed=False)
    accounting = _property_accounting(failures.NOT_ACCOUNTABLE, complete_case=-0.01)
    accounting.loc[1, ["worst_case", "imputation"]] = [math.nan, "complete case only"]
    unaccountable = accounting.iloc[[0]].assign(
        leg=property_failures.NOT_ACCOUNTABLE_LEG,
        complete_case=math.nan,
        worst_case=math.nan,
        imputation="rejection ceiling",
        leg_passed=False,
    )
    tables["property_accounting"] = pd.concat(
        [accounting.iloc[[0]], unaccountable, accounting.iloc[[1]]], ignore_index=True
    )
    (row,) = red_rows(RECORD, tables)
    assert row.fails_by == "its own rule"
    # The mutation control: the same cell green at its complete case reads the accounting.
    tables["property_accounting"].loc[2, ["complete_case", "leg_passed"]] = [0.01, True]
    (row,) = red_rows(RECORD, tables)
    assert row.fails_by == "failure accounting: failures not accountable"


def test_a_failed_truth_row_is_red() -> None:
    (row,) = red_rows(RECORD, _synthetic(truth_passed=False))
    assert row.key == RedKey(RECORD.slug, "truth", "cleverly/law/ate")
    assert row.role == "cleverly"


def test_an_accepted_reference_failure_is_not_red() -> None:
    # The renamed reference declares no bound of its own; the bound does not enter this test.
    record = dataclasses.replace(
        RECORD,
        reference="reference",
        accepted_reference_failure="declared in advance",
        failure_bound={},
    )
    frames_ = _synthetic(truth_passed=False, implementation="reference")
    frames_["performance"] = pd.concat(
        [_synthetic()["performance"], frames_["performance"]], ignore_index=True
    )
    assert red_rows(record, frames_) == []
    undeclared = dataclasses.replace(record, accepted_reference_failure="")
    assert [row.key.kind for row in red_rows(undeclared, frames_)] == ["truth"]


# --- deliberate mutations: the findings ------------------------------------------------------


KEY = RedKey("study", "property", "family/cell")
ROW = RedRow(KEY, "positive", "its own rule", "bias 0 to 1")


def test_findings_refuse_a_red_row_in_a_gated_study() -> None:
    (finding,) = findings([ROW], {KEY: "RM18-x"}, {"study": "gated"}, ["RM18-x"])
    assert "registered as 'gated'" in finding


def test_findings_refuse_an_unowned_red_row() -> None:
    (finding,) = findings([ROW], {}, {"study": "reporting"}, ["RM18-x"])
    assert "has no owner" in finding


def test_findings_refuse_an_owner_that_is_not_red() -> None:
    (finding,) = findings([], {KEY: "RM18-x"}, {"study": "reporting"}, ["RM18-x"])
    assert "is not red" in finding


def test_findings_refuse_an_owner_the_roadmap_does_not_list() -> None:
    (finding,) = findings([ROW], {KEY: "RM18-x"}, {"study": "reporting"}, ["RM18-y"])
    assert "does not list" in finding


def test_a_row_claimed_twice_is_refused() -> None:
    with pytest.raises(ValueError, match="claimed by"):
        owners({"RM18-x": (KEY,), "RM18-y": (KEY,)})


def test_the_accepted_row_reads_as_accepted_in_its_table_and_no_other_row_does() -> None:
    """The exception changes one cell's label and nothing else in any accuracy table."""
    from tests.studies.evidence.document import ACCEPTED_LABEL, accuracy_table

    assert ACCEPTED_LABEL == "accepted (named exception)"
    labelled = []
    for study in STUDIES:
        data = frames(study)
        frame = data["performance"]
        lines = accuracy_table(study, data)[2:]
        ordered = frame.sort_values(["scenario", "estimand", "implementation"])
        for row, line in zip(ordered.itertuples(), lines, strict=True):
            cell = line.rstrip(" |").rsplit(" | ", 1)[1]
            key = RedKey(study.slug, "truth", f"{row.implementation}/{row.scenario}/{row.estimand}")
            if key in ACCEPTED_BY_EXCEPTION and key not in OWNERS:
                assert not row.passed
                assert cell == ACCEPTED_LABEL
                labelled.append(key)
            else:
                assert cell == ("pass" if row.passed else "**fail**")
    assert labelled == [ACCEPTED_KEY]
