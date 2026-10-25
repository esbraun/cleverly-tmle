"""The combined reports print like every other check (X89 T3).

The contract test (``test_summary_contract.py``) checks width, notes, house style and every
printed row of each fixture in ``summary_fixtures/battery.py``. This module checks what is
particular to the combined reports: the title follows the facade, a shared name names its
surface, the reprs, that the contract's code mask is load-bearing, and that the row reader
fails a renderer that moves rows.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any

import pytest

from cleverly._assessment_summary import check_name
from cleverly._inference_status import STATUS_RECORDS
from cleverly._summary import SummaryTable, _render, vocabulary
from cleverly.assessment import AssessmentStatus

from .summary_fixtures import battery
from .summary_fixtures._battery_reader import read_battery

CASES = battery.FIXTURES


@functools.cache
def _report(name: str) -> Any:
    return CASES[name].build()


def test_the_title_follows_the_facade_that_ran() -> None:
    assert _report("battery_diagnostics").surface == "diagnostics"
    assert _report("battery_sensitivity").surface == "sensitivity"
    assert (
        _report("battery_sensitivity").summary().heading.strip()
        == "Sensitivity analyses of the fit"
    )
    assert _report("battery_diagnostics").summary().heading.strip() == "Diagnostics of the fit"
    assert _report("battery_validation").summary().heading.strip() == "Quick checks of the fit"


def test_a_name_two_surfaces_share_names_its_surface() -> None:
    summary = _report("battery_every_status").summary()
    labels = [row[0].text for table in summary.tables for row in table.rows]
    assert "equations solved (validation)" in labels
    assert "equations solved (sensitivity)" in labels
    assert "overlap" in labels


def test_every_status_prints_in_words_and_the_result_follows_the_worst() -> None:
    summary = _report("battery_every_status").summary()
    [checks] = [table for table in summary.tables if table.title == "Checks"]
    words = {row[1].text for row in checks.rows}
    assert words == {"failed", "needs attention", "passed"}
    assert summary.facts[vocabulary.RESULT] == "failed"
    failed = _report("battery_failed_validation").summary()
    assert failed.facts[vocabulary.RESULT] == "failed"
    assert failed.reading == "equations solved failed."
    passed = _report("battery_validation")
    assert passed.passed
    assert passed.summary().facts[vocabulary.RESULT] == "passed"


def test_a_deferred_validation_row_makes_the_result_incomplete() -> None:
    report = _report("battery_failed_validation")
    deferred = dataclasses.replace(report, items=report.items[1:])
    summary = deferred.summary()
    assert summary.facts[vocabulary.RESULT] == "incomplete"
    assert summary.reading == (
        "outcome and treatment models did not run, so validation is incomplete."
    )
    assert not deferred.passed


def test_the_full_detail_prints_each_reason_whole() -> None:
    report = _report("battery_every_status")
    lines = report.summary("full").section(vocabulary.BATTERY_ROWS_IN_FULL).lines
    assert "refutation tests (on request): not run by default; pass a flag" in lines
    short = str(report.summary())
    assert "pass a flag" not in short.split("Notes:")[0]


def test_a_deferred_refit_says_what_it_costs() -> None:
    report = _report("battery_diagnostics")
    refute = report["refute"]
    assert refute.status is AssessmentStatus.DEFERRED
    assert refute.detail.startswith("not run by default because it refits the models")


def test_the_code_mask_is_load_bearing() -> None:
    """Witness: the next steps print a public call whose name holds an internal term."""
    note = _report("battery_every_status").summary().note("next_steps").text
    assert "result.diagnostics.nuisance_models()" in note
    assert "nuisance" in note.lower()
    assert "nuisance" not in vocabulary.PUBLIC_CALL.sub("CALL", note).lower()


def test_a_labelled_interval_adds_its_plain_reason() -> None:
    summary = _report("battery_collaborative").summary()
    reason = STATUS_RECORDS["working_mechanism_plugin"].plain_reason
    assert summary.note("inference_status").text == f"About the interval: {reason}"
    assert "inference_status" not in _report("battery_assessment").summary().note_codes


def test_an_unknown_operation_prints_with_spaces() -> None:
    assert check_name("new_check") == "new check"
    assert check_name("support") == "overlap"


def test_the_reprs_are_one_line() -> None:
    assert repr(_report("battery_every_status")) == "AssessmentReport(8 rows, 2 need attention)"
    assert repr(_report("battery_failed_validation")) == "ValidationReport(2 rows, failed)"
    assert repr(_report("battery_diagnostics")).startswith("DiagnosticReport(")
    item = _report("battery_every_status").validation.items[0]
    assert repr(item) == "AssessmentItem(score_equations, failed)"
    assert _report("battery_assessment")._repr_html_().startswith("<")


# ----------------------------------------------------------------- mutation controls

_ORIGINAL = _render._table_lines


def _swap_rows(table: SummaryTable, width: int) -> Any:
    """The renderer with the label of the first two rows of every table swapped."""
    if len(table.rows) >= 2:
        first, second, *rest = table.rows
        rows = ((second[0], *first[1:]), (first[0], *second[1:]), *rest)
        table = dataclasses.replace(table, rows=tuple(rows))
    return _ORIGINAL(table, width)


@pytest.mark.parametrize(
    "name", ["battery_every_status", "battery_assessment", "battery_sensitivity"]
)
def test_a_renderer_that_swaps_two_rows_fails_the_reader(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _report(name)
    read_battery(report, str(report.summary()))
    monkeypatch.setattr(_render, "_table_lines", _swap_rows)
    with pytest.raises(AssertionError):
        read_battery(report, str(report.summary()))
