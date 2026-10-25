"""The outcome and treatment model check prints like every other check (X89 T3).

The contract test (``test_summary_contract.py``) checks width, notes, the repr and every
printed number of each fixture in ``summary_fixtures/models.py``. This module checks what is
particular to the model check: that each warning and note fires where its rule fires and
only there, that the warnings' numbers equal the check's, that the reading and ``verdict()``
are one sentence, and that the number reader fails a renderer that moves numbers between
rows or columns.
"""

from __future__ import annotations

import dataclasses
import functools
import re
import warnings
from pathlib import Path
from typing import Any

import pytest

from cleverly._summary import SummaryTable, _render, vocabulary
from cleverly.validation._nuisance_summary import model_words

from .summary_fixtures import models
from .summary_fixtures._check_fidelity import read_models

CHECKS = dict(models.FIXTURES)


@functools.cache
def _check(name: str) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CHECKS[name]()


def test_every_note_of_the_check_fires() -> None:
    fired = {code for name in CHECKS for code in _check(name).summary().note_codes}
    assert set(vocabulary.T3_MODEL_NOTE_CODES) <= fired, set(vocabulary.T3_MODEL_NOTE_CODES) - fired


def test_no_section_redeclares_a_note_code() -> None:
    """Each section rebinds ``NOTE_CODES`` with ``{**NOTE_CODES, **SECTION}``, so a code that
    two sections declare silently takes the later text everywhere. The model check's
    ``collaborative_model`` once took the name of the point-estimate ``working_model``
    note and changed what a longitudinal working-model fit printed."""
    source = Path(vocabulary.__file__).read_text()
    declared = re.findall(r'^\s+"(\w+)": "(?:note|warning)",?$', source, re.MULTILINE)
    twice = sorted({code for code in declared if declared.count(code) > 1})
    assert not twice, twice
    assert set(declared) == set(vocabulary.NOTE_CODES)


def test_the_reading_is_the_verdict_and_counts_the_findings() -> None:
    for name in CHECKS:
        check = _check(name)
        summary = check.summary()
        assert check.verdict() == summary.reading
        count = len(check.findings)
        warned = [item for item in summary.notes if item.kind == "warning"]
        assert len(warned) == count, (name, summary.note_codes)
        if count:
            assert dict(summary.header)[vocabulary.RESULT] == "needs attention"
            assert summary.reading.startswith(f"{count} finding")
        else:
            assert dict(summary.header)[vocabulary.RESULT] == "no pass or fail"
            assert summary.reading == vocabulary.MODEL_READINGS["clear"]


def test_the_findings_keep_their_sentences_as_data() -> None:
    check = _check("models_findings")
    assert check.findings[0] == (
        "the propensity model separates the arms almost perfectly (AUC 0.950); this signals "
        "a positivity problem, not a good fit"
    )
    assert check.findings[1].startswith("propensity put 90% of its weight on the marginal mean")
    assert check.findings[2].startswith("outcome: the out-of-fold predictions are more extreme")


def test_the_warnings_carry_the_checks_numbers() -> None:
    summary = _check("models_findings").summary()
    assert "(AUC 0.950)" in summary.note("separation").text
    assert "put 90% of its weight" in summary.note("mean_only").text
    calibration = summary.note("calibration_finding").text
    assert "calibration slope 0.40" in calibration
    assert "more extreme" in calibration and "Estimation noise" in calibration
    assert (
        "(R squared 0.010)"
        in _check("models_weak_outcome").summary().note("weak_outcome_model").text
    )
    assert "(AUC 0.970)" in _check("models_recording").summary().note("recording_separation").text


def test_the_weak_treatment_note_is_a_note_and_skips_a_working_model() -> None:
    weak = _check("models_weak_treatment")
    summary = weak.summary()
    assert "weak_treatment_model" in summary.note_codes
    assert weak.findings == ()
    assert "(AUC 0.520)" in summary.note("weak_treatment_model").text
    # Witness: the collaborative fit's working model is barely predictive too, and stays
    # quiet because its AUC describes the model the search chose.
    collaborative = _check("models_collaborative")
    assert collaborative["propensity"].metrics["auc"] < 0.55
    assert "weak_treatment_model" not in collaborative.summary().note_codes
    ordinary = dataclasses.replace(collaborative, treatment_role="estimated_treatment_law")
    assert "weak_treatment_model" in ordinary.summary().note_codes


def test_the_calibration_note_follows_the_evaluation() -> None:
    out_of_fold = _check("models_cross_fitted").summary().note("calibration").text
    assert "one intercept per fold" in out_of_fold and "Bonferroni" in out_of_fold
    in_sample = _check("models_in_sample").summary().note("calibration").text
    assert "in sample and carry no test" in in_sample
    assert "Bonferroni" not in in_sample


def test_a_repeated_fit_names_its_split_and_prints_the_spread() -> None:
    summary = _check("models_repeated").summary()
    assert dict(summary.header)[vocabulary.SPLIT_SHOWN] == "split 1 of 3"
    titles = [table.title for table in summary.tables]
    assert vocabulary.SPREAD_TABLE in titles
    assert "split_ratio" in summary.note_codes
    single = _check("models_cross_fitted").summary()
    assert vocabulary.SPLIT_SHOWN not in dict(single.header)
    assert "split_ratio" not in single.note_codes


def test_the_full_detail_states_what_the_fit_could_not_report() -> None:
    lines = _check("models_missing_parts").summary("full").section(vocabulary.SECTION_DETAILS)
    assert lines.lines == (
        vocabulary.MODEL_OMISSIONS["selection"].format(
            reason="the fitted result retains no ctmle method artifact"
        ),
        vocabulary.MODEL_OMISSIONS["outcome"].format(
            reason="this estimator fits no outcome regression"
        ),
    )
    plain = _check("models_cross_fitted").summary("full").section(vocabulary.SECTION_DETAILS)
    assert plain.lines == (vocabulary.MODEL_NOTHING_OMITTED,)


@pytest.mark.parametrize(
    ("name", "words"),
    [
        ("propensity", "treatment model"),
        ("propensity[placebo]", "treatment model (arm placebo)"),
        ("outcome", "outcome model"),
        ("missingness", "recording model"),
        ("treatment_observation[1]", "treatment-recording model (arm 1)"),
        ("intermediate", "intermediate model"),
        ("something_new", "something_new"),
    ],
)
def test_model_names_read_in_plain_words(name: str, words: str) -> None:
    assert model_words(name) == words


def test_the_reprs_are_one_line() -> None:
    check = _check("models_cross_fitted")
    assert repr(check) == "NuisanceDiagnostics(2 models, out of fold, no pass or fail)"
    propensity = check["propensity"]
    assert repr(propensity) == (
        f"NuisanceModelReport(treatment model, AUC={propensity.metrics['auc']:.3f}, "
        f"Brier score={propensity.metrics['brier']:.3f})"
    )


# ----------------------------------------------------------------- mutation controls

_ORIGINAL = _render._table_lines


def _swap_rows(table: SummaryTable, width: int) -> Any:
    """The renderer with the numbers of the first two rows of every table swapped."""
    if len(table.rows) >= 2:
        first, second, *rest = table.rows
        rows = ((first[0], *second[1:]), (second[0], *first[1:]), *rest)
        table = dataclasses.replace(table, rows=tuple(rows))
    return _ORIGINAL(table, width)


def _differ(table: SummaryTable) -> bool:
    if len(table.rows) < 2:
        return False
    first, second = table.rows[:2]
    return [cell.value for cell in first[1:]] != [cell.value for cell in second[1:]]


SWAPPABLE = sorted(
    name for name in CHECKS if any(_differ(table) for table in _check(name).summary().tables)
)


@pytest.mark.parametrize("name", SWAPPABLE)
def test_a_renderer_that_swaps_two_rows_fails_the_reader(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    check = _check(name)
    read_models(check, str(check.summary()))
    monkeypatch.setattr(_render, "_table_lines", _swap_rows)
    with pytest.raises(AssertionError):
        read_models(check, str(check.summary()))


def test_the_row_swap_reaches_most_fixtures() -> None:
    assert len(SWAPPABLE) >= len(CHECKS) // 2, SWAPPABLE


@pytest.mark.parametrize("name", sorted(set(CHECKS) - {"models_none"}))
def test_a_renderer_that_swaps_two_metric_columns_fails_the_reader(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    check = _check(name)
    pairs = (("AUC", "Brier score"), ("R2", "MSE"))

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
        return _ORIGINAL(table, width)

    monkeypatch.setattr(_render, "_table_lines", swapped)
    with pytest.raises(AssertionError):
        read_models(check, str(check.summary()))
