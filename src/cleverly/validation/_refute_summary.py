"""The printed summary of a refutation report (X89 T3).

A refutation report prints like every other result: a header of facts, a table with one
row per test, a table of the statistics behind each reading, and one note per test that
says what the test does, what a reading can and cannot show, and whether the test gives a
pass or fail for this kind of fit. The numbers are read from the report; nothing here
computes a test or decides a verdict.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .._inference_status import supplies_inference
from .._summary import Cell, Note, Summary, SummarySection, SummaryTable, short_repr, vocabulary
from .._summary._estimates import without_roadmap
from .._summary._render import decimals_for, number_text, pvalue_text
from ..inference.influence import spread_name

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .refute import RefutationResult, RefutationTest

__all__ = ["refutation_summary", "report_repr", "test_repr"]

_NUMBER_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


def _count(value: int) -> str:
    return _NUMBER_WORDS[value] if 0 <= value < len(_NUMBER_WORDS) else f"{value:,}"


def _joined(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _name(test: RefutationTest) -> str:
    return vocabulary.REFUTATION_NAMES.get(test.name, test.name.replace("_", " "))


def _result(test: RefutationTest) -> str:
    key = {"passed": "passed", "failed": "failed"}.get(test.verdict, "descriptive")
    return vocabulary.CHECK_RESULTS[key]


def _null(test: RefutationTest) -> str:
    """The value a test expects with no effect, as the purpose sentence names it."""
    expectation = test.expectation.strip()
    if expectation.startswith("~ "):
        return expectation[2:]
    return "the value it has with no effect"


def _expected(test: RefutationTest) -> str:
    """The ``expected`` cell: today's expectation with its symbols read out."""
    text = test.expectation.strip()
    if text.startswith("readings inside their bands around "):
        return f"about {text.rsplit(' ', 1)[-1]}, see bands"
    if text.startswith("spread ") and text.endswith(" around the original fit"):
        return f"SD about {text.split()[1]}"
    if text.startswith("moves from "):
        return f"moves from {text.split()[2]}"
    for symbol, words in (
        ("log-scale scatter ~ ", "log SD about "),
        ("scatter ~ ", "SD about "),
        ("~ ", "about "),
    ):
        if text.startswith(symbol):
            return words + text[len(symbol) :]
    return text


def _not_run(test: RefutationTest) -> bool:
    return not test.values and test.declaration is None


def _purpose(test: RefutationTest) -> str:
    key = "placebo_not_run" if test.name == "placebo" and _not_run(test) else test.name
    template = vocabulary.REFUTATION_PURPOSES.get(key)
    if template is None:
        return f"cleverly runs the {_name(test)} test."
    variables = getattr(test.declaration, "variables", ())
    words = [vocabulary.READING_WORDS[reading.name] for reading in test.readings]
    return template.format(
        null=_null(test),
        variables=", ".join(map(str, variables)),
        readings=_joined(words) if words else "the refits",
    )


def _evidence(test: RefutationTest) -> str:
    """The note's last clause: whether the test gives a pass or fail for this fit, and why."""
    words = vocabulary.EVIDENCE_WORDS
    if _not_run(test):
        return words["not_run"]
    if test.passed is not None:
        return words[test.rung]
    if test.withheld:
        return words["withheld"].format(reason=test.withheld)
    if test.rung in ("L2", "L3"):
        return words["studied"]
    return words.get(test.rung, words["L0"])


def _test_note(test: RefutationTest) -> Note:
    conditions = " ".join(without_roadmap(item) for item in test.conditions if item)
    evidence = _evidence(test)
    text = vocabulary.NOTE_TEXTS["refutation_test"].format(
        name=_name(test).capitalize(),
        purpose=_purpose(test),
        conditions=f" {conditions}" if conditions else "",
        evidence=evidence,
    )
    return Note(f"test_{test.name}", text, "warning" if test.passed is False else "note")


def _finite(value: float) -> float | None:
    """A cell value: ``None`` for a missing number, so two equal tables compare equal."""
    return value if math.isfinite(value) else None


def _decimals(report: RefutationResult) -> int:
    spreads = [test.spread for test in report.tests if math.isfinite(test.spread)]
    return decimals_for(spreads)


def _tests_table(report: RefutationResult, decimals: int) -> SummaryTable:
    rows = tuple(
        (
            Cell(test.name, _name(test)),
            Cell(_finite(test.mean), number_text(test.mean, decimals)),
            Cell(_finite(test.spread), number_text(test.spread, decimals)),
            Cell(test.expectation, _expected(test)),
            Cell(test.verdict, _result(test)),
        )
        for test in report.tests
    )
    return SummaryTable(("test", "mean", "SD", "expected", "result"), rows)


def _statistic_label(test: RefutationTest, label: str) -> str:
    plain = vocabulary.STATISTIC_NAMES.get(label, label)
    null = test.expectation[2:] if test.expectation.startswith("~ ") else "its null"
    return plain.format(null=null)


def _statistics_table(report: RefutationResult) -> SummaryTable | None:
    rows: list[tuple[Cell, ...]] = []
    for test in report.tests:
        if test.statistic is not None:
            rows.append(
                (
                    Cell(test.name, _name(test)),
                    Cell(test.statistic_name, _statistic_label(test, test.statistic_name)),
                    Cell(test.statistic, number_text(test.statistic, 3)),
                    Cell(test.statistic_pvalue, pvalue_text(test.statistic_pvalue)),
                )
            )
        for label, value in test.extras:
            rows.append(
                (
                    Cell(test.name, _name(test)),
                    Cell(label, _statistic_label(test, label)),
                    Cell(value, number_text(value, 3)),
                    Cell(None, vocabulary.MISSING),
                )
            )
    if not rows:
        return None
    return SummaryTable(
        ("test", "statistic", "value", "p-value"),
        tuple(rows),
        title=vocabulary.STATISTICS_TITLE,
    )


def _readings_table(report: RefutationResult) -> SummaryTable | None:
    rows: list[tuple[Cell, ...]] = []
    for test in report.tests:
        for reading in test.readings:
            low, high = reading.band
            rows.append(
                (
                    Cell(test.name, _name(test)),
                    Cell(reading.name, _statistic_label(test, reading.name)),
                    Cell(reading.estimate, number_text(reading.estimate, 3)),
                    Cell(
                        (reading.lower, reading.upper),
                        f"[{number_text(reading.lower, 3)}, {number_text(reading.upper, 3)}]",
                    ),
                    Cell((low, high), f"[{low:g}, {high:g}]"),
                    Cell(reading.inside, "yes" if reading.inside else "no"),
                )
            )
    if not rows:
        return None
    return SummaryTable(
        ("test", "reading", "estimate", "95% CI", "allowed band", "inside"),
        tuple(rows),
        title=vocabulary.READINGS_TITLE,
    )


def _reading(report: RefutationResult) -> str:
    verdict = report.passed
    if verdict is True:
        return vocabulary.REFUTATION_READINGS["passed"]
    if verdict is False:
        failed = [_name(test) for test in report.tests if test.passed is False]
        return vocabulary.REFUTATION_READINGS["failed"].format(names=_joined(failed).capitalize())
    open_tests = sum(test.passed is None for test in report.tests)
    if len(report.tests) == 1:
        return vocabulary.REFUTATION_READINGS["descriptive_one"]
    if open_tests == len(report.tests) == 2:
        return vocabulary.REFUTATION_READINGS["descriptive_both"]
    if open_tests == len(report.tests):
        return vocabulary.REFUTATION_READINGS["descriptive_all"].format(n=_count(len(report.tests)))
    return vocabulary.REFUTATION_READINGS["descriptive_some"].format(
        k=_count(open_tests).capitalize(),
        n=_count(len(report.tests)),
        verb="gives" if open_tests == 1 else "give",
    )


def _notes(report: RefutationResult) -> list[Note]:
    notes = [_test_note(test) for test in report.tests]
    failures = [test for test in report.tests if test.n_failed]
    if failures:
        notes.append(
            Note(
                "refit_failures",
                " ".join(
                    vocabulary.NOTE_TEXTS["refit_failures"].format(
                        failed=test.n_failed,
                        total=test.n_replicates,
                        name=_name(test),
                        test=test.name,
                    )
                    for test in failures
                ),
                "warning",
            )
        )
    groups: dict[str, list[str]] = {}
    for test in report.tests:
        if not supplies_inference(test.inference):
            label = vocabulary.SPREAD_LABELS[spread_name("std_err", test.inference)]
            groups.setdefault(label, []).append(_name(test))
    if groups:
        text = " ".join(
            vocabulary.NOTE_TEXTS["refit_spread_diagnostic"].format(
                names=_joined(names), label=label
            )
            for label, names in groups.items()
        )
        notes.append(Note("refit_spread_diagnostic", text, "warning"))
    geometric = [_name(test) for test in report.tests if test.mean_kind == "geometric"]
    logged = [_name(test) for test in report.tests if test.spread_kind == "log"]
    parts = [
        template.format(names=_joined(names) + (" test" if len(names) == 1 else " tests"))
        for template, names in (
            (vocabulary.RATIO_GEOMETRIC, geometric),
            (vocabulary.RATIO_LOG, logged),
        )
        if names
    ]
    if parts:
        text = vocabulary.NOTE_TEXTS["ratio_scale"].format(parts=", and ".join(parts))
        notes.append(Note("ratio_scale", text, "note"))
    return notes


def _sections(report: RefutationResult) -> list[SummarySection]:
    compared = tuple(
        f"{_name(test).capitalize()}: {test.evidence or vocabulary.EVIDENCE_WORDS['L0']}"
        for test in report.tests
    )
    sections = [SummarySection(vocabulary.SECTION_COMPARED, compared)]
    draws: list[str] = []
    for test in report.tests:
        if test.declaration is None:
            count = len(test.values)
            draws.append(f"{_name(test).capitalize()}: {count:,} refit{'s' * (count != 1)}.")
            continue
        line = (
            f"{_name(test).capitalize()}: {len(test.records):,} refits succeeded and "
            f"{test.n_failed:,} failed"
        )
        variables = getattr(test.declaration, "variables", None)
        if variables:
            line += f"; error added to {', '.join(map(str, variables))}"
        if test.resampling is not None:
            line += f"; {vocabulary.RESAMPLING_WORDS.get(test.resampling, test.resampling)}"
        elif getattr(test.declaration, "name", None):
            name = test.declaration.name
            line += f"; outcome drawn as {vocabulary.PROCESS_NAMES.get(name, name)}"
            if test.declared_effect is not None and math.isfinite(test.declared_effect):
                line += f", with true effect {test.declared_effect:.4g}"
        draws.append(line + ".")
    sections.append(SummarySection(vocabulary.SECTION_DRAWS, tuple(draws)))
    return sections


def refutation_summary(report: RefutationResult, detail: str) -> Summary:
    """The summary of a refutation report.

    Parameters
    ----------
    report : RefutationResult
        The report.
    detail : {"short", "full"}
        How much to print.

    Returns
    -------
    Summary
        The summary.
    """
    decimals = _decimals(report)
    original = report.tests[0].original if report.tests else float("nan")
    repeats = sorted({test.n_replicates for test in report.tests})
    header = [
        (vocabulary.ESTIMATE_TESTED, f"{report.estimand} = {number_text(original, decimals)}"),
        (vocabulary.TESTS_RUN, f"{len(report.tests):,}"),
        (
            vocabulary.REPEATS_PER_TEST,
            f"{repeats[0]:,}" if len(repeats) == 1 else f"{repeats[0]:,} to {repeats[-1]:,}",
        ),
        (vocabulary.RESULT, _report_result(report)),
    ]
    tables = [_tests_table(report, decimals)]
    tables.extend(table for table in (_statistics_table(report), _readings_table(report)) if table)
    return Summary.build(
        title=vocabulary.REFUTATION_TITLE,
        header=header,
        header_split=2,
        tables=tables,
        reading=_reading(report),
        notes=_notes(report),
        sections=_sections(report) if detail == "full" else (),
    )


def _report_result(report: RefutationResult) -> str:
    verdict = report.passed
    key = "descriptive" if verdict is None else ("passed" if verdict else "failed")
    return vocabulary.CHECK_RESULTS[key]


def test_repr(test: RefutationTest) -> str:
    """The one-line repr of a refutation test.

    Parameters
    ----------
    test : RefutationTest
        The test.

    Returns
    -------
    str
        For example ``RefutationTest(placebo, mean=-0.045, sd=0.091, no pass or fail)``.
    """
    return short_repr(
        "RefutationTest",
        [
            ("", test.name),
            ("mean", f"{test.mean:.3g}"),
            ("sd", f"{test.spread:.3g}"),
            ("", _result(test)),
        ],
    )


def report_repr(report: RefutationResult) -> str:
    """The one-line repr of a refutation report.

    Parameters
    ----------
    report : RefutationResult
        The report.

    Returns
    -------
    str
        For example ``RefutationResult(ate, 3 tests, no pass or fail)``.
    """
    return short_repr(
        "RefutationResult",
        [
            ("", report.estimand),
            ("", f"{len(report.tests)} test{'s' * (len(report.tests) != 1)}"),
            ("", _report_result(report)),
        ],
    )
