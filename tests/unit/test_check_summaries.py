"""The printed summaries of refutation reports (X89 T3).

``test_summary_contract.py`` runs the shared contract over every fixture in
``summary_fixtures/checks.py``. These tests cover what is particular to a refutation
report: that every note and reading fires, that the number reader catches a renderer that
moves numbers, and that the plain words agree with the records they describe.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import warnings
from typing import Any

import pytest

from cleverly._summary import SummaryTable, _render, vocabulary
from cleverly.validation import _interpretation, outcome_processes
from cleverly.validation._interpretation import CalibrationScope, FitConfiguration
from cleverly.validation.refute import RefutationResult, refute

from .summary_fixtures import checks
from .summary_fixtures._check_fidelity import read_refutation

REPORTS = dict(checks.FIXTURES)


@functools.cache
def _report(name: str) -> RefutationResult:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return REPORTS[name]()


#: Test notes no fixture here fires: the future-treatment test needs a longitudinal fit.
UNFIRED_TEST_NOTES = {"test_future_treatment"}


def test_every_refutation_note_fires() -> None:
    codes = {f"test_{name}" for name in vocabulary.REFUTATION_NAMES}
    codes |= {"refit_failures", "refit_spread_diagnostic", "ratio_scale"}
    assert codes <= set(vocabulary.NOTE_CODES)
    fired = {code for name in REPORTS for code in _report(name).summary().note_codes}
    assert fired | UNFIRED_TEST_NOTES >= codes, codes - fired - UNFIRED_TEST_NOTES
    assert not UNFIRED_TEST_NOTES & fired, "a fixture fires it now; drop it from the list"


def test_every_reading_fires() -> None:
    readings = {_report(name).summary().reading for name in REPORTS}
    for key in ("passed", "descriptive_one"):
        assert vocabulary.REFUTATION_READINGS[key] in readings, key
    assert vocabulary.REFUTATION_READINGS["descriptive_all"].format(n="three") in readings
    assert any(reading.startswith("Shuffled treatment failed.") for reading in readings)
    some = _report("refute_failed")
    partly = dataclasses.replace(
        some, tests=(dataclasses.replace(some.tests[0], passed=True), *some.tests[1:])
    )
    assert partly.summary().reading.startswith("No test failed. Two of the three tests give")


def test_a_failed_test_is_a_warning_and_its_result_says_failed() -> None:
    summary = _report("refute_failed").summary()
    assert summary.note("test_placebo").kind == "warning"
    assert summary.note("test_subset").kind == "note"
    assert dict(summary.header)[vocabulary.RESULT] == vocabulary.CHECK_RESULTS["failed"]
    assert [cell.text for cell in summary.tables[0].rows[0]][-1] == "failed"
    assert (
        "2 of 40 repeats of the measurement error test failed"
        in summary.note("refit_failures").text
    )


def test_a_note_says_what_the_test_can_show_for_this_fit() -> None:
    words = vocabulary.EVIDENCE_WORDS
    passed = _report("refute_passed").summary().note("test_placebo").text
    assert passed.endswith(words["L2"] + ".")
    open_ = _report("refute_default").summary().note("test_placebo").text
    assert open_.endswith(words["L0"] + ".")
    generated = _report("refute_generated")
    simulated = generated["simulated_outcome"]
    assert simulated.rung == "L1" and simulated.withheld
    text = generated.summary().note("test_simulated_outcome").text
    assert text.endswith(
        words["withheld"].format(reason="its 10 repeats are fewer than the 400 its study used")
        + "."
    )
    assert words["L1"] not in text


def test_a_studied_test_without_a_verdict_never_claims_one() -> None:
    """A licensed rung with no verdict on this report reads as studied, not as a pass or
    fail, and its result column agrees."""
    report = _report("refute_default")
    placebo = dataclasses.replace(report["placebo"], rung="L2", passed=None)
    summary = dataclasses.replace(report, tests=(placebo,)).summary()
    text = summary.note("test_placebo").text
    assert text.endswith(vocabulary.EVIDENCE_WORDS["studied"] + ".")
    assert vocabulary.EVIDENCE_WORDS["L2"] not in text
    assert summary.tables[0].rows[0][-1].text == vocabulary.CHECK_RESULTS["descriptive"]


def test_a_placebo_that_ran_no_refit_says_so() -> None:
    report = _report("refute_default")
    placebo = dataclasses.replace(report["placebo"], values=(), expectation="no exact value")
    text = dataclasses.replace(report, tests=(placebo,)).summary().note("test_placebo").text
    assert "runs no refit here" in text
    assert text.endswith(vocabulary.EVIDENCE_WORDS["not_run"] + ".")
    assert "should be about" not in text


def test_the_numbers_in_the_notes_and_sections_equal_the_report() -> None:
    report = _report("refute_failed")
    error = report["bootstrap_measurement_error"]
    text = report.summary().note("refit_failures").text
    assert f"{error.n_failed} of {error.n_replicates} repeats" in text
    lines = report.summary(detail="full").section(vocabulary.SECTION_DRAWS).lines
    assert any(
        f"{len(error.records):,} refits succeeded and {error.n_failed:,} failed" in line
        for line in lines
    )
    assert any(f"{len(report['placebo'].values):,} refits." in line for line in lines)
    generated = _report("refute_generated")
    lines = generated.summary(detail="full").section(vocabulary.SECTION_DRAWS).lines
    for test in generated.tests:
        assert any(f"true effect {test.declared_effect:.4g}" in line for line in lines)


def test_a_report_with_no_test_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one refutation test"):
        refute(checks._studied(), tests=())


def test_a_ratio_names_which_tests_use_which_scale() -> None:
    text = _report("refute_ratio").summary().note("ratio_scale").text
    assert "the mean of the shuffled treatment and added noise column tests is the geometric" in (
        text
    )
    assert "the SD of the random subsamples test is on the log scale" in text


def test_the_generated_law_prints_in_plain_words_in_the_full_summary() -> None:
    report = _report("refute_generated")
    for test in report.tests:
        assert "Generated law: " in test.scope_sentence
        assert not any(item.startswith("Generated law: ") for item in test.conditions)
    lines = report.summary(detail="full").section(vocabulary.SECTION_DRAWS).lines
    assert any("Gaussian noise unrelated to the data, with true effect 0" in line for line in lines)


def test_every_outcome_process_has_a_plain_name() -> None:
    names = {
        field.default
        for _, cls in inspect.getmembers(outcome_processes, inspect.isclass)
        if dataclasses.is_dataclass(cls)
        for field in dataclasses.fields(cls)
        if field.name == "name" and isinstance(field.default, str)
    }
    assert len(names) >= 6
    assert names <= set(vocabulary.PROCESS_NAMES), names - set(vocabulary.PROCESS_NAMES)


# ----------------------------------------------------------------- the number reader


def _swap_rows(table: SummaryTable, width: int) -> Any:
    """The renderer with the numbers of the first two rows of every table swapped."""
    if len(table.rows) >= 2:
        first, second, *rest = table.rows
        rows = ((first[0], *second[1:]), (second[0], *first[1:]), *rest)
        table = dataclasses.replace(table, rows=tuple(rows))
    return _ORIGINAL(table, width)


_ORIGINAL = _render._table_lines


#: The fixtures with a table of two or more rows, which a row swap can move.
SWAPPABLE = sorted(
    name for name in REPORTS if name not in {"refute_negative_control", "refute_measurement_error"}
)


@pytest.mark.parametrize("name", SWAPPABLE)
def test_a_renderer_that_swaps_two_rows_fails_the_reader(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _report(name)
    read_refutation(report, str(report.summary()))
    monkeypatch.setattr(_render, "_table_lines", _swap_rows)
    with pytest.raises(AssertionError):
        read_refutation(report, str(report.summary()))


def test_every_swappable_fixture_has_two_rows_and_the_rest_one() -> None:
    """Witness: the row swap reaches every fixture it lists, and only those skip it."""
    for name in REPORTS:
        assert (len(_report(name).tests) >= 2) is (name in SWAPPABLE), name


@pytest.mark.parametrize("name", sorted(REPORTS))
def test_a_renderer_that_swaps_mean_and_sd_fails_the_reader(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _report(name)
    original = _render._table_lines

    def swapped(table: SummaryTable, width: int) -> Any:
        if table.columns[:3] == ("test", "mean", "SD"):
            rows = tuple((row[0], row[2], row[1], *row[3:]) for row in table.rows)
            table = dataclasses.replace(table, rows=rows)
        return original(table, width)

    monkeypatch.setattr(_render, "_table_lines", swapped)
    with pytest.raises(AssertionError):
        read_refutation(report, str(report.summary()))


def test_the_reader_fails_a_table_it_does_not_know(monkeypatch: pytest.MonkeyPatch) -> None:
    report = _report("refute_default")
    text = str(report.summary()).replace(vocabulary.STATISTICS_TITLE, "Other numbers")
    with pytest.raises(AssertionError, match="no reader"):
        read_refutation(report, text)


# ----------------------------------------------------------------- plain scope words

#: For each axis :meth:`CalibrationScope.mismatches` names, a word its plain phrase holds.
AXIS_WORDS = (
    ("method ", "another method"),
    ("estimand kind ", "another estimate"),
    ("outcome family ", "another kind of outcome"),
    ("treatment kind ", "another kind of treatment"),
    ("cross-fitting ", "out-of-fold"),
    ("inference status ", "interval"),
    ("known mechanism ", "treatment model"),
    ("arm count ", "treatment levels"),
    ("replicate count ", "repeats"),
    ("node count ", "time points"),
    ("placebo draw ", "shuffled treatment"),
    ("learner ", "learner"),
    ("n = ", "rows"),
    ("covariates per row ", "covariates"),
    ("minimum event count ", "events"),
    ("cluster count ", "clusters"),
)


def _fit(**changes: Any) -> FitConfiguration:
    base = FitConfiguration(
        method="TMLE",
        estimand_kind="arm:ate",
        outcome_family="gaussian",
        treatment_kind="discrete",
        learners=("linear",),
        cross_fitting="folds=5",
        inference="influence_curve",
        known_mechanism=False,
        n=5_000,
        covariates_per_row=0.001,
        min_event_count=None,
        clusters=None,
        arms=2,
    )
    return dataclasses.replace(base, **changes)


def _scopes() -> list[CalibrationScope]:
    rows = _interpretation.VERDICT_EVIDENCE
    return [CalibrationScope.unrestricted(), *(row.scope for row in rows)]


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"method": "IPTW", "n": 50},
        {"cross_fitting": "off", "inference": "estimated_mechanism_plugin"},
        {"inference": "few_cluster_plugin", "clusters": 4},
        {"known_mechanism": True, "outcome_family": "binomial", "n": 10_000_000},
        {"estimand_kind": "arm:rr", "arms": 3, "learners": ("forest",), "treatment_kind": "dose"},
        {"replicates": 7, "nodes": 3, "placebo_draw": "markov", "min_event_count": 2},
        {"covariates_per_row": 0.9},
    ],
)
def test_plain_mismatches_name_the_same_axes_as_mismatches(changes: dict[str, Any]) -> None:
    fit = _fit(**changes)
    reached: set[str] = set()
    for scope in _scopes():
        axes = scope.mismatches(fit)
        plain = _interpretation.plain_mismatches(scope, fit)
        assert len(plain) == len(axes), (scope, plain, axes)
        for axis, phrase in zip(axes, plain, strict=True):
            prefix, word = next(item for item in AXIS_WORDS if axis.startswith(item[0]))
            assert word in phrase, (axis, phrase)
            reached.add(prefix)
    assert reached or not changes


def test_every_axis_reaches_the_plain_phrase_check() -> None:
    """Witness: the cases above make every axis mismatch at least once."""
    cases = test_plain_mismatches_name_the_same_axes_as_mismatches.pytestmark[0].args[1]
    reached = {
        next(prefix for prefix, _ in AXIS_WORDS if axis.startswith(prefix))
        for changes in cases
        for scope in _scopes()
        for axis in scope.mismatches(_fit(**changes))
    }
    assert reached == {prefix for prefix, _ in AXIS_WORDS}, {
        prefix for prefix, _ in AXIS_WORDS
    } - reached


def test_an_interval_from_a_plugin_status_is_not_called_no_interval() -> None:
    scope = CalibrationScope(inference=frozenset({"influence_curve"}))
    plain = _interpretation.plain_mismatches(scope, _fit(inference="few_cluster_plugin"))
    assert plain == ["an interval from a plug-in standard error"]
    plain = _interpretation.plain_mismatches(scope, _fit(inference="estimated_mechanism_plugin"))
    assert plain == ["no interval"]


def test_an_unreadable_fit_and_a_scope_print_plainly() -> None:
    scope = CalibrationScope(n_min=2_000)
    assert _interpretation.plain_mismatches(scope, None) == ["a fit cleverly cannot read"]
    assert _interpretation.plain_mismatches(CalibrationScope.unrestricted(), None) == []
    assert scope.mismatches(None) and not CalibrationScope.unrestricted().mismatches(None)
    assert "2,000 rows or more" in _interpretation.plain_scope(scope)
    both = CalibrationScope(
        inference=frozenset({"estimated_mechanism_plugin", "generated_design_plugin"})
    )
    assert "no interval or no interval" not in _interpretation.plain_scope(both)
