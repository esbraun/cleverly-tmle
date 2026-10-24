"""The equation check and the DR-TMLE correction check print like every other check (X89 T3).

The contract test (``test_summary_contract.py``) checks width, notes, the repr and every
printed number of each fixture in ``summary_fixtures/equations.py``. This module checks
what is particular to these two checks: that every note and reading fires, that the notes'
numbers equal the check's, that the plain names cover every equation a fit can print, and
that the number readers fail a renderer that moves numbers between rows or columns.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import re
import warnings
from typing import Any

import pytest

from cleverly._summary import SummaryTable, _render, vocabulary
from cleverly.validation._score_summary import equation_words

from .summary_fixtures import equations
from .summary_fixtures._check_fidelity import (
    read_corrections,
    read_equations,
    sci_close,
    text_tables,
)

CHECKS = dict(equations.FIXTURES)
SCORES = sorted(name for name in CHECKS if name.startswith("score_"))
CORRECTIONS = sorted(name for name in CHECKS if name.startswith("corrections_"))


@functools.cache
def _check(name: str) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CHECKS[name]()


def _reader(name: str) -> Any:
    return read_equations if name in SCORES else read_corrections


def test_every_note_of_the_two_checks_fires() -> None:
    codes = set(vocabulary.T3_EQUATION_NOTE_CODES) | {"stopping_rule"}
    fired = {code for name in CHECKS for code in _check(name).summary().note_codes}
    assert codes <= fired, codes - fired


def _pattern(template: str) -> re.Pattern[str]:
    """A template as a pattern, each placeholder matching any text."""
    parts = re.split(r"\{[^}]*\}", template)
    return re.compile(".+?".join(re.escape(part) for part in parts), re.S)


def test_every_reading_fires() -> None:
    readings = [_check(name).summary().reading for name in CHECKS]
    templates = {
        **vocabulary.EQUATION_READINGS,
        **{f"correction_{key}": value for key, value in vocabulary.CORRECTION_READINGS.items()},
    }
    for key, template in templates.items():
        pattern = _pattern(template)
        assert any(pattern.search(reading) for reading in readings), key


def test_a_failed_row_is_a_warning_and_the_result_says_failed() -> None:
    summary = _check("score_failed").summary()
    assert summary.facts[vocabulary.RESULT] == vocabulary.CHECK_RESULTS["failed"]
    kinds = {note.code: note.kind for note in summary.notes}
    assert kinds["identity_failure"] == kinds["row_not_converged"] == "warning"
    # The pair: a passing check prints neither.
    passing = _check("score_drtmle").summary()
    assert passing.facts[vocabulary.RESULT] == vocabulary.CHECK_RESULTS["passed"]
    assert not {"identity_failure", "row_not_converged"} & set(passing.note_codes)


def test_the_two_curve_notes_never_print_together() -> None:
    for name in SCORES:
        codes = set(_check(name).summary().note_codes)
        assert not {"equations_only", "dr_curve"} <= codes, name


def test_a_check_without_corrections_has_no_pass_or_fail() -> None:
    summary = _check("corrections_none").summary()
    assert summary.facts[vocabulary.RESULT] == vocabulary.CHECK_RESULTS["descriptive"]
    assert summary.reading == vocabulary.CORRECTION_READINGS["none"]
    assert [[cell.text for cell in row] for row in summary.tables[0].rows] == [["none reported"]]


def test_the_numbers_in_the_notes_equal_the_check() -> None:
    check = _check("corrections_in_sample")
    text = check.summary().note("correction_scale").text
    assert f"{check.scale:.4g} times" in text
    shown = re.findall(r"\d\.\d+e[-+]\d+", text)
    assert sci_close(shown[0], check.threshold) and sci_close(shown[1], check.identity_threshold)
    # A check on the fitting scale has no scale sentence.
    flat = dataclasses.replace(check, scale=1.0).summary().note("correction_scale").text
    assert "times the scale" not in flat

    failed = _check("score_failed")
    row = next(row for row in failed.rows if row.failure)
    text = failed.summary().note("row_not_converged").text
    assert f"condition number {row.hessian_condition:.2e}" in text
    good, total = row.folds_converged
    assert f"{total - good:,} of {total} folds" in text


@pytest.mark.parametrize("fit", [equations.declared_fit, equations.declared_drtmle_fit])
def test_the_rule_note_is_the_estimate_summarys_note(fit: Any) -> None:
    """One remainder, one note: the check states it as the estimate summary does, down to a
    remainder below the smallest printed one (the DR-TMLE fit's)."""
    result = fit()
    note = result.summary().note("stopping_rule").text
    assert result.diagnostics.score_equations().summary().note("stopping_rule").text == note


def test_a_rule_remainder_below_the_smallest_printed_one_says_so() -> None:
    check = _check("score_declared_drtmle")
    worst = max(row.residual_se for row in check.rows if math.isfinite(row.residual_se))
    assert worst < 0.001
    text = check.summary().note("stopping_rule").text
    assert "less than 0.001 standard errors" in text
    assert "95.00%" not in text


def test_the_details_give_each_step_its_error_before_and_after() -> None:
    check = _check("score_tmle")
    lines = check.summary(detail="full").section(vocabulary.SECTION_DETAILS).lines
    steps = [row for row in check.rows if math.isfinite(row.score_initial)]
    assert len(lines) == len(steps) > 0
    for line, row in zip(lines, steps, strict=True):
        before, after = re.findall(r"\d\.\d+e[-+]\d+", line)
        assert sci_close(before, abs(row.score_initial), row.threshold), line
        assert sci_close(after, abs(row.score), row.threshold), line


def test_the_correction_details_give_each_split_its_margins() -> None:
    check = _check("corrections_splits")
    lines = check.summary(detail="full").section(vocabulary.SECTION_DETAILS).lines
    assert [line.split(":")[0] for line in lines] == ["Split 1", "Split 2"]
    margin = min(row.margin for row in check.rows if row.draw == 0)
    assert f"within {margin:.2g} of its limits" in lines[0]


@pytest.mark.parametrize(
    ("name", "words"),
    [
        ("mean", "outcome means"),
        ("att", "att"),
        ("mean (mechanism)", "outcome means (treatment model)"),
        ("mean (reduced)[draw 1]", "outcome means (reduced regression, split 2)"),
        ("msm (strata)", "projection (strata)"),
        ("mean (D*_g)[1]", "outcome means (treatment correction, arm 1)"),
        ("mean (D*_Q)[0] identity[draw 0]", "outcome means (outcome correction, arm 0, split 1)"),
        ("mean (D*_M)[treated]", "outcome means (recording correction, arm treated)"),
        ("mean (D*_Y)[1]", "outcome means (outcome correction, arm 1)"),
        ("mean (D*_A)[0]", "outcome means (treatment correction, arm 0)"),
    ],
)
def test_equation_names_read_in_plain_words(name: str, words: str) -> None:
    assert equation_words(name) == words


def test_no_summary_prints_a_correction_symbol() -> None:
    for name in CHECKS:
        text = str(_check(name).summary(detail="full"))
        assert "D*" not in text and "g_r1" not in text and "pi*" not in text, name


# ----------------------------------------------------------------- the number readers


def _swap_rows(table: SummaryTable, width: int) -> Any:
    """The renderer with the numbers of the first two rows of every table swapped."""
    if len(table.rows) >= 2:
        first, second, *rest = table.rows
        rows = ((first[0], *second[1:]), (second[0], *first[1:]), *rest)
        table = dataclasses.replace(table, rows=tuple(rows))
    return _ORIGINAL(table, width)


_ORIGINAL = _render._table_lines


def _differ(table: SummaryTable) -> bool:
    """Whether a table's first two rows print different cells after the row label."""
    if len(table.rows) < 2:
        return False
    first, second = table.rows[:2]
    return [cell.text for cell in first[1:]] != [cell.text for cell in second[1:]]


#: The fixtures where a row swap moves a printed number.
SWAPPABLE = sorted(
    name for name in CHECKS if any(_differ(table) for table in _check(name).summary().tables)
)


@pytest.mark.parametrize("name", SWAPPABLE)
def test_a_renderer_that_swaps_two_rows_fails_the_reader(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    check = _check(name)
    _reader(name)(check, str(check.summary()))
    monkeypatch.setattr(_render, "_table_lines", _swap_rows)
    with pytest.raises(AssertionError):
        _reader(name)(check, str(check.summary()))


def test_the_row_swap_reaches_every_check_family() -> None:
    """Witness: at least one fixture of each check moves a number under the swap."""
    assert set(SWAPPABLE) & set(SCORES), SWAPPABLE
    assert set(SWAPPABLE) & set(CORRECTIONS), SWAPPABLE
    assert len(SWAPPABLE) >= len(CHECKS) // 2, SWAPPABLE


@pytest.mark.parametrize("name", sorted(set(CHECKS) - {"corrections_none", "score_none"}))
def test_a_renderer_that_swaps_two_number_columns_fails_the_reader(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    check = _check(name)
    original = _render._table_lines
    pairs = {("error", "allowed"), ("in SE", "cap bias")}

    def swapped(table: SummaryTable, width: int) -> Any:
        for left, right in pairs:
            if left in table.columns and right in table.columns:
                i, j = table.columns.index(left), table.columns.index(right)
                rows = []
                for row in table.rows:
                    cells = list(row)
                    cells[i], cells[j] = cells[j], cells[i]
                    rows.append(tuple(cells))
                table = dataclasses.replace(table, rows=tuple(rows))
        return original(table, width)

    monkeypatch.setattr(_render, "_table_lines", swapped)
    with pytest.raises(AssertionError):
        _reader(name)(check, str(check.summary()))


#: The fixtures with a printed error far from zero, which a zeroed error must fail.
NONZERO = sorted(
    name
    for name in CHECKS
    if any(
        label in ("error", "in SE") and cell.value and abs(cell.value) > 1e-9
        for table in _check(name).summary().tables
        for row in table.rows
        for label, cell in zip(table.columns, row, strict=True)
    )
)


@pytest.mark.parametrize("name", NONZERO)
def test_a_renderer_that_prints_every_error_as_zero_fails_the_reader(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    check = _check(name)
    original = _render._table_lines

    def zeroed(table: SummaryTable, width: int) -> Any:
        columns = [i for i, label in enumerate(table.columns) if label in ("error", "in SE")]
        rows = tuple(
            tuple(
                dataclasses.replace(cell, text="0.000e+00") if i in columns else cell
                for i, cell in enumerate(row)
            )
            for row in table.rows
        )
        return original(dataclasses.replace(table, rows=rows), width)

    monkeypatch.setattr(_render, "_table_lines", zeroed)
    with pytest.raises(AssertionError):
        _reader(name)(check, str(check.summary()))


def test_the_zeroed_error_reaches_both_check_families() -> None:
    assert set(NONZERO) & set(SCORES) and set(NONZERO) & set(CORRECTIONS), NONZERO


def test_a_renderer_that_renames_a_correction_fails_the_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _render._table_lines

    def renamed(table: SummaryTable, width: int) -> Any:
        if table.columns[0] == "correction" and table.rows:
            first, *rest = table.rows
            cell = first[0]
            other = "outcome" if cell.text != "outcome" else "treatment"
            first = (dataclasses.replace(cell, text=other), *first[1:])
            table = dataclasses.replace(table, rows=(first, *rest))
        return original(table, width)

    monkeypatch.setattr(_render, "_table_lines", renamed)
    for name in ("score_drtmle", "corrections_in_sample"):
        check = _check(name)
        with pytest.raises(AssertionError):
            _reader(name)(check, str(check.summary()))


def test_an_arm_label_with_a_bracket_prints_whole() -> None:
    check = _check("score_drtmle")
    labels = {row.label: f"{row.label}]x" for row in check.corrections.rows}
    corrections = dataclasses.replace(
        check.corrections,
        rows=tuple(
            dataclasses.replace(row, label=labels[row.label]) for row in check.corrections.rows
        ),
    )
    names = {}
    for old, new in zip(check.corrections.rows, corrections.rows, strict=True):
        names[old.name] = new.name
        names[f"{old.name} identity"] = f"{new.name} identity"
    rows = tuple(dataclasses.replace(row, name=names.get(row.name, row.name)) for row in check.rows)
    summary = dataclasses.replace(check, rows=rows, corrections=corrections).summary()
    arms = {cell.text for cell in summary.tables[1].column("arm")}
    assert arms == set(labels.values())
    assert "D*" not in summary


def test_the_reader_fails_a_table_it_does_not_know() -> None:
    check = _check("score_tmle")
    text = str(check.summary()).replace(vocabulary.ESTIMATE_EQUATIONS_TITLE, "Other numbers")
    with pytest.raises(AssertionError):
        read_equations(check, text)


def test_a_wrapped_row_is_read_back_whole() -> None:
    """The reader joins a row name that wrapped at the margin and a last column of words
    that wrapped indented, so a long cell stays checked."""
    text = "\n".join(
        [
            "Title",
            "=" * 20,
            "name        value  result",
            "-" * 20,
            "a long row  1.000  no pass or",
            "            fail",
            "another     2.000  passed",
            "row name",
            "=" * 20,
        ]
    )
    (_, labels, rows), *_ = text_tables(text)
    assert labels == ["name", "value", "result"]
    assert rows == [
        ["a long row", "1.000", "no pass or fail"],
        ["another row name", "2.000", "passed"],
    ]
