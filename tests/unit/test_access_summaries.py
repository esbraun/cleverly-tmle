"""The facades, their declared checks and the replay record print in words (X89 T3).

``repr(result.diagnostics)`` lists the checks a fit can run, its notebook display lists every
declared check with its question, whether it can run and its cost, and a declared check and a
replay record each read as one line. ``Replayability.summary()`` joins the contract fixtures in
``summary_fixtures/replay.py``; this module checks the rest.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

import cleverly.longitudinal.estimator as longitudinal_estimator
from cleverly import assessment
from cleverly._assessment_summary import facade_repr, facade_summary, replay_part
from cleverly._summary import SummaryTable, _render, vocabulary
from cleverly.assessment import AssessmentStatus, Replayability

from .summary_fixtures import replay
from .summary_fixtures.battery import _fit


def test_a_facade_repr_lists_the_checks_it_can_run() -> None:
    result = _fit()
    text = repr(result.diagnostics)
    available = [row.operation for row in result.diagnostics.capabilities if row.available]
    assert text == f"DiagnosticsFacade({', '.join(available)})"
    assert "corrections" not in text


def test_a_long_facade_repr_is_cut_at_100_characters() -> None:
    result = _fit()
    text = repr(result.sensitivity)
    assert len(text) <= 100
    assert text.endswith(", ...)")
    assert text.startswith("SensitivityFacade(omitted_confounding, ")
    rows = result.sensitivity.capabilities
    whole = facade_repr("SensitivityFacade", [row for row in rows if row.available][:2])
    assert not whole.endswith("...)")


def test_the_facade_display_lists_every_declared_check() -> None:
    result = _fit()
    rows = result.sensitivity.capabilities
    summary = facade_summary("sensitivity", rows)
    assert summary.heading.strip() == "Sensitivity analyses available for this fit"
    [table] = summary.tables
    assert [row[0].text for row in table.rows] == [row.operation for row in rows]
    by_name = {row[0].text: row for row in table.rows}
    assert by_name["evalue"][1].text == "E-value"
    assert by_name["marginal_bounds"][3].text == "refits the models"
    assert by_name["evalue"][3].text == "cheap"
    assert by_name["missingness"][2].text == "no"
    note = summary.note("why_not").text
    assert note.startswith("Cannot run here: missingness (does not apply)")
    assert summary.facts[vocabulary.CHECKS_AVAILABLE] == str(sum(row.available for row in rows))
    assert result.diagnostics._repr_html_().startswith("<")
    assert all(len(line) <= 78 for line in str(summary).splitlines())


def test_a_fit_where_every_check_runs_adds_no_note() -> None:
    rows = [row for row in _fit().diagnostics.capabilities if row.available]
    assert "why_not" not in facade_summary("diagnostics", rows).note_codes


def test_a_declared_check_reads_as_one_line() -> None:
    result = _fit()
    refute = result.diagnostics.capability("refute")
    assert repr(refute) == "AssessmentCapability(refute, available, refits the models)"
    support = result.diagnostics.capability("support")
    assert repr(support) == "AssessmentCapability(support, available, cheap)"
    corrections = result.diagnostics.capability("corrections")
    assert corrections.status is AssessmentStatus.NOT_APPLICABLE
    assert repr(corrections) == "AssessmentCapability(corrections, does not apply)"


def test_a_replay_record_names_what_it_can_redo() -> None:
    assert repr(replay.point()) == "Replayability(summarize, re-estimate, refit, rerun)"
    assert repr(Replayability(False, False, False, False, False)) == "Replayability(nothing)"
    assert replay.point()._repr_html_().startswith("<")


def test_the_reading_names_what_a_record_cannot_redo() -> None:
    assert replay.everything().summary().reading == "This result can redo every action listed."
    summary = replay.every_part().summary()
    assert summary.reading == (
        "This result cannot evaluate other quantities from the stored weighting function, "
        "refit the models, score new data or rerun the same settings on changed data."
    )
    none = Replayability(False, False, False, False, False).summary()
    assert none.reading == "This result can redo none of the actions listed."


def test_every_missing_part_prints_in_words() -> None:
    note = replay.every_part().summary().note("missing_parts").text
    assert note.startswith("The result lacks the estimator's settings, ")
    assert "_" not in note
    assert "missing_parts" not in replay.everything().summary().note_codes


def test_every_replay_code_has_plain_words() -> None:
    """A new code fails here rather than printing with underscores."""
    codes = {
        value
        for module in (assessment, longitudinal_estimator)
        for name, value in vars(module).items()
        if name.startswith(("POINT_REPLAY_", "LONGITUDINAL_REPLAY_")) and isinstance(value, str)
    }
    assert len(codes) >= 8
    assert codes <= set(vocabulary.REPLAY_PARTS)
    assert replay_part("estimator configuration") == "the estimator's settings"
    assert replay_part("a_new_code") == "a new code"


def test_the_full_detail_says_what_each_action_needs() -> None:
    lines = replay.point().summary("full").section(vocabulary.REPLAY_NEEDS).lines
    assert len(lines) == 6
    assert lines[3].startswith("refit the models: the estimator's settings")


# ----------------------------------------------------------------- mutation control

_ORIGINAL = _render._table_lines


def _swap_cells(table: SummaryTable, width: int) -> Any:
    """The renderer with the answers of the first and third rows swapped."""
    first, second, third, *rest = table.rows
    rows = ((first[0], third[1]), second, (third[0], first[1]), *rest)
    return _ORIGINAL(dataclasses.replace(table, rows=rows), width)


@pytest.mark.parametrize("name", ["point", "every_part"])
def test_a_renderer_that_moves_an_answer_fails_the_reader(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = getattr(replay, name)()
    replay.read_replay(record, str(record.summary()))
    monkeypatch.setattr(_render, "_table_lines", _swap_cells)
    with pytest.raises(AssertionError):
        replay.read_replay(record, str(record.summary()))
