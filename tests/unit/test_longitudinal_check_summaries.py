"""The three longitudinal checks print like every other check (X89 T3).

The contract test (``test_summary_contract.py``) checks width, notes, the repr and every
printed number of each fixture in ``summary_fixtures/longitudinal_checks.py``. This module
checks what is particular to these checks: that each note fires where its rule fires and
only there, that the Result and the reading follow the rows, that the full detail carries
the SDR, IPSI, step and metric values, and that the number readers fail a renderer that moves
numbers between rows or columns.
"""

from __future__ import annotations

import dataclasses
import functools
import warnings
from typing import Any

import pytest

from cleverly._summary import SummaryTable, _render, vocabulary
from cleverly.validation._longitudinal_summary import node_words
from cleverly.validation.longitudinal import LongitudinalNuisanceOmission

from .summary_fixtures import longitudinal_checks
from .summary_fixtures._check_fidelity import (
    read_longitudinal_equations,
    read_longitudinal_models,
    read_longitudinal_overlap,
)

CASES = longitudinal_checks.FIXTURES


@functools.cache
def _check(name: str) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CASES[name].build()


def _codes(name: str) -> set[str]:
    return set(_check(name).summary().note_codes)


def test_every_note_of_the_checks_fires() -> None:
    fired = set().union(*(_codes(name) for name in CASES))
    declared = set(vocabulary.T3_LONGITUDINAL_NOTE_CODES)
    assert declared <= fired, declared - fired
    assert {"first_split", "row_not_converged", "split_ratio"} <= fired


def test_the_overlap_reading_names_the_thinnest_time_point() -> None:
    summary = _check("overlap_means").summary()
    assert summary.reading == (
        "Overlap is thinnest for never at time 2: 44.2 effective rows of 61 followed."
    )
    competing = _check("overlap_competing").summary().reading
    assert competing.startswith("Overlap is thinnest for never (cause relapse, horizon 2)")


def test_a_capped_time_point_and_another_splits_failure_are_warnings() -> None:
    summary = _check("overlap_capped_witness").summary()
    assert summary.facts[vocabulary.RESULT] == "needs attention"
    capped = summary.note("capped_nodes").text
    assert capped.startswith("At never at time 2 the lower limit replaced")
    assert capped.endswith("The limit replaced every row of some time point in split 3.")
    assert summary.note("not_converged").text == (
        "A correction step also failed to converge in split 2."
    )


def test_a_share_just_below_every_row_is_no_capped_time_point() -> None:
    """Witness: the warning reads a share of one, not a large share."""
    check = _check("overlap_capped_witness")
    rows = list(check.rows)
    rows[-1] = dataclasses.replace(rows[-1], share_truncated=0.999)
    moved = dataclasses.replace(
        check, rows=tuple(rows), draw_max_share_truncated=(0.999, 0.0, 0.999)
    )
    assert "capped_nodes" not in moved.summary().note_codes


def test_an_unconverged_step_adds_the_column_and_the_warning() -> None:
    summary = _check("overlap_unsolved").summary()
    assert summary.facts[vocabulary.RESULT] == "needs attention"
    [table] = summary.tables
    assert vocabulary.STAGE_CONVERGED in table.columns
    assert summary.note("not_converged").text.startswith(
        "The correction step did not converge at never at time 2."
    )
    plain = _check("overlap_means").summary()
    assert plain.facts[vocabulary.RESULT] == "no pass or fail"
    assert vocabulary.STAGE_CONVERGED not in plain.tables[0].columns


def test_policy_and_tilt_time_points_say_what_assigned_holds() -> None:
    policy = _check("overlap_policy").summary()
    assert policy.note("policy_nodes").text.startswith("At mix at time 1 and mix at time 2")
    assert policy.note("assigned").text == vocabulary.ASSIGNED_SHARES
    assert "tilt_nodes" in _codes("overlap_ipsi")
    assert "policy_nodes" not in _codes("overlap_means")


def test_the_full_detail_carries_the_sdr_and_ipsi_measures() -> None:
    check = _check("overlap_ipsi")
    lines = check.summary("full").section(vocabulary.SECTION_DETAILS).lines
    first = check.rows[0]
    assert lines[0].startswith(
        f"natural course at time 1: {first.sdr.target_outside:.3f} of the regression targets"
    )
    tilt = check.rows[2].ipsi
    assert lines[len(check.rows) + 2].startswith(
        f"odds x2 at time 1: the tilt reweights a unit by a factor from "
        f"{tilt.tilt_factor_min:.3f} to {tilt.tilt_factor_max:.3f}"
    )
    assert len(lines) == 2 * len(check.rows)
    sdr = _check("overlap_sdr").summary("full").section(vocabulary.SECTION_DETAILS).lines
    assert len(sdr) == len(_check("overlap_sdr").rows)
    plain = _check("overlap_means").summary("full").section(vocabulary.SECTION_DETAILS)
    assert plain.lines == (vocabulary.OVERLAP_NOTHING_MORE,)


def test_a_failed_equation_names_the_time_point_and_the_reason() -> None:
    summary = _check("equations_unsolved").summary()
    assert summary.facts[vocabulary.RESULT] == "failed"
    assert summary.reading == (
        "1 of 4 equations was not solved (never at time 2), so the standard error may be wrong."
    )
    assert summary.note("row_not_converged").text == (
        "never at time 2: the correction step stopped because most predictions sat at their bounds."
    )
    passed = _check("equations_means").summary()
    assert passed.facts[vocabulary.RESULT] == "passed"
    assert passed.reading.startswith("The correction step solved every estimating equation")


def test_an_unconverged_row_without_a_reason_still_warns() -> None:
    check = _check("equations_unsolved")
    rows = tuple(dataclasses.replace(row, failure=None) for row in check.rows)
    note = dataclasses.replace(check, rows=rows).summary().note("row_not_converged")
    assert note.text == "never at time 2: the correction step did not converge."


def test_working_model_and_split_rows_say_what_they_are() -> None:
    pooled = _check("equations_working_model").summary()
    assert "working_model_rows" in pooled.note_codes
    assert pooled.tables[0].columns[:2] == ("time", "component")
    assert "every_split" in _codes("equations_repeats")
    assert _check("equations_repeats").summary().tables[0].columns[0] == "split"


def test_the_full_detail_lists_each_equations_steps() -> None:
    check = _check("equations_unsolved")
    lines = check.summary("full").section(vocabulary.SECTION_DETAILS).lines
    assert lines[0] == vocabulary.EQUATION_SCALE
    last = check.rows[-1]
    assert lines[-1] == (f"never at time 2: {last.n_iter} step, equation value {last.score:.3e}.")


def test_the_models_name_each_omission_and_each_metric() -> None:
    summary = _check("models_ipw").summary()
    assert summary.note("model_omitted").text == (
        "No outcome model: inverse probability weighting fits no outcome regression."
    )
    check = _check("models_means")
    lines = check.summary("full").section(vocabulary.SECTION_DETAILS).lines
    treatment = check.rows[0].model.metrics
    assert lines[0] == (
        f"every regimen at time 1, treatment model: AUC {treatment['auc']:.3f}, "
        f"calibration slope {treatment['calibration_slope']:.3f}."
    )
    iterated = next(row for row in check.rows if row.role == "pseudo_outcome")
    assert f"always at time 1, iterated outcome model: R2 {iterated.model.metrics['r2']:.3f}." in (
        lines
    )


def test_an_omission_names_its_time_point_regimen_and_folds() -> None:
    check = _check("models_means")
    omission = LongitudinalNuisanceOmission(
        "pseudo_outcome", 2, "a reason", regimen="always", folds=(1, 3)
    )
    text = dataclasses.replace(check, omissions=(omission,)).summary().note("model_omitted").text
    assert text == "No iterated outcome (always) model at time 2 in fold 1 and 3: a reason."


@pytest.mark.parametrize(
    ("arguments", "words"),
    [
        ({"regimen": "always", "time": 2}, "always at time 2"),
        (
            {"regimen": None, "time": 1, "component": "dose"},
            ("every regimen (working-model term dose) at time 1"),
        ),
        (
            {"regimen": "never", "time": 1, "cause": "death", "horizon": 2, "draw": 3},
            "never (cause death, horizon 2, split 3) at time 1",
        ),
    ],
)
def test_a_row_reads_in_words(arguments: dict[str, Any], words: str) -> None:
    assert node_words(**arguments) == words


def test_the_reprs_are_one_line() -> None:
    assert repr(_check("overlap_unsolved")) == "LongitudinalDiagnostics(4 rows, needs attention)"
    assert repr(_check("equations_unsolved")) == "LongitudinalScoreDiagnostics(4 rows, failed)"
    assert repr(_check("models_means")) == (
        "LongitudinalNuisanceDiagnostics(8 rows, no pass or fail)"
    )
    assert _check("models_means")._repr_html_().startswith("<")


# ----------------------------------------------------------------- mutation controls

_ORIGINAL = _render._table_lines

_READERS = {
    "overlap": read_longitudinal_overlap,
    "equations": read_longitudinal_equations,
    "models": read_longitudinal_models,
}


def _reader(name: str) -> Any:
    return _READERS[name.split("_", 1)[0]]


def _swap_rows(table: SummaryTable, width: int) -> Any:
    """The renderer with the cells of the first two rows of every table swapped."""
    if len(table.rows) >= 2:
        first, second, *rest = table.rows
        rows = ((first[0], *second[1:]), (second[0], *first[1:]), *rest)
        table = dataclasses.replace(table, rows=tuple(rows))
    return _ORIGINAL(table, width)


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_renderer_that_swaps_two_rows_fails_the_reader(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    check = _check(name)
    first, second = check.summary().tables[0].rows[:2]
    assert [cell.value for cell in first[1:]] != [cell.value for cell in second[1:]]
    _reader(name)(check, str(check.summary()))
    monkeypatch.setattr(_render, "_table_lines", _swap_rows)
    with pytest.raises(AssertionError):
        _reader(name)(check, str(check.summary()))


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_renderer_that_swaps_two_number_columns_fails_the_reader(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    check = _check(name)
    pairs = (("max weight", "eff. rows"), ("rows", "loss"), ("time", "remaining error"))

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
                break
        return _ORIGINAL(table, width)

    monkeypatch.setattr(_render, "_table_lines", swapped)
    # A count cell holding a decimal fails to parse, which fails the reader as well.
    with pytest.raises((AssertionError, ValueError)):
        _reader(name)(check, str(check.summary()))
