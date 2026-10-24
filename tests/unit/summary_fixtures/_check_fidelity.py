"""Readers that check every number a check or refutation summary prints (X89 T3).

A reader parses the printed text, not the ``Summary`` parts, and compares each number in the
header and the tables with the value read independently from the object: its ``to_frame()``
or its attributes, never a ``Cell.value``. A table the reader does not know fails, so a new
table cannot slip through unread. Numbers inside notes and full-detail sections are checked
by ``tests/unit/test_check_summaries.py``.
"""

from __future__ import annotations

import math
import re
from typing import Any

from cleverly._summary import vocabulary

__all__ = ["header_values", "read_refutation", "text_tables"]

_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?:e[-+]?\d+)?(?![\w])")


def _rule(line: str, char: str) -> bool:
    return bool(line) and set(line) == {char}


def header_values(text: str) -> dict[str, str]:
    """The header facts of a printed summary, label to printed value.

    Parameters
    ----------
    text : str
        The printed summary.

    Returns
    -------
    dict of str to str
        One entry per header fact.
    """
    lines = text.splitlines()
    first = next(index for index, line in enumerate(lines) if _rule(line, "="))
    facts: dict[str, str] = {}
    for line in lines[first + 1 :]:
        if _rule(line, "="):
            break
        for part in re.split(r"\s{3,}(?=[A-Z][^:]*:)", line.strip()):
            label, _, value = part.partition(":")
            facts[label.strip()] = value.strip()
    return facts


def text_tables(text: str) -> list[tuple[str, list[str], list[list[str]]]]:
    """Every table of a printed summary: its title, its column labels and its rows.

    Parameters
    ----------
    text : str
        The printed summary.

    Returns
    -------
    list of tuple
        ``(title, labels, rows)``, the title empty when the table has none. A row is the
        list of its cells, split on runs of two or more spaces.
    """
    lines = text.splitlines()
    found = []
    for index, line in enumerate(lines):
        if not _rule(line, "-") or index == 0:
            continue
        labels = re.split(r"\s{2,}", lines[index - 1].strip())
        title = lines[index - 2].strip() if index >= 2 and not _rule(lines[index - 2], "=") else ""
        rows = []
        for row in lines[index + 1 :]:
            if _rule(row, "=") or not row.strip():
                break
            rows.append(re.split(r"\s{2,}", row.strip()))
        found.append((title, labels, rows))
    return found


def _decimals(text: str) -> int:
    return len(text.split(".")[1]) if "." in text else 0


def close(shown: str, value: float | None) -> bool:
    """Whether a printed number agrees with ``value`` to half a unit of its last place.

    Parameters
    ----------
    shown : str
        The printed number; ``"-"`` matches a missing or non-finite value.
    value : float or None
        The value read from the object.

    Returns
    -------
    bool
        Whether the two agree.
    """
    missing = value is None or (isinstance(value, float) and not math.isfinite(value))
    if shown == vocabulary.MISSING:
        return missing
    if missing:
        return False
    if shown.startswith("<"):
        return float(value) < float(shown[1:])
    return (
        abs(float(shown.replace(",", "")) - float(value)) <= 0.5 * 10 ** -_decimals(shown) + 1e-12
    )


def _numbers(text: str) -> list[str]:
    return _NUMBER.findall(text)


def _test_names(report: Any) -> dict[str, Any]:
    return {
        vocabulary.REFUTATION_NAMES.get(test.name, test.name.replace("_", " ")): test
        for test in report.tests
    }


_RESULTS = {
    "passed": vocabulary.CHECK_RESULTS["passed"],
    "failed": vocabulary.CHECK_RESULTS["failed"],
    "descriptive": vocabulary.CHECK_RESULTS["descriptive"],
}


def read_refutation(report: Any, text: str) -> None:
    """Assert that every number a refutation summary prints equals the report's.

    Parameters
    ----------
    report : RefutationResult
        The report.
    text : str
        Its short summary.
    """
    frame = report.to_frame().set_index("test")
    names = _test_names(report)
    facts = header_values(text)
    estimand, _, shown = facts[vocabulary.ESTIMATE_TESTED].partition(" = ")
    assert estimand == report.estimand, facts
    for value in frame["original"]:
        assert close(shown, float(value)), (shown, value)
    assert int(facts[vocabulary.TESTS_RUN].replace(",", "")) == len(frame)
    repeats = [int(item.replace(",", "")) for item in _numbers(facts[vocabulary.REPEATS_PER_TEST])]
    assert repeats == sorted({min(repeats), max(repeats)})
    counts = {test.n_replicates for test in report.tests}
    assert (min(repeats), max(repeats)) == (min(counts), max(counts)), (repeats, counts)
    verdict = report.passed
    expected = "descriptive" if verdict is None else ("passed" if verdict else "failed")
    assert facts[vocabulary.RESULT] == _RESULTS[expected]

    tables = text_tables(text)
    assert tables, "the summary printed no table"
    _, labels, rows = tables[0]
    assert labels == ["test", "mean", "SD", "expected", "result"], labels
    assert [row[0] for row in rows] == list(names), rows
    for row in rows:
        test = names[row[0]]
        record = frame.loc[test.name]
        assert close(row[1], record["refuted_mean"]), (row, record["refuted_mean"])
        assert close(row[2], record["refuted_sd"]), (row, record["refuted_sd"])
        promised = _numbers(str(record["expectation"]))
        for number in _numbers(row[3]):
            assert any(close(number, float(item.replace(",", ""))) for item in promised), (
                row,
                record["expectation"],
            )
        assert row[4] == _RESULTS[str(record["verdict"])], (row, record["verdict"])

    for title, labels, rows in tables[1:]:
        if title == vocabulary.STATISTICS_TITLE:
            _read_statistics(report, frame, names, labels, rows)
        elif title == vocabulary.READINGS_TITLE:
            _read_readings(names, labels, rows)
        else:
            raise AssertionError(f"no reader for the table {title!r}")


def _read_statistics(
    report: Any, frame: Any, names: dict[str, Any], labels: list[str], rows: list[list[str]]
) -> None:
    assert labels == ["test", "statistic", "value", "p-value"], labels
    expected: list[tuple[str, str, float, float | None]] = []
    for test in report.tests:
        record = frame.loc[test.name]
        statistic = record["statistic"]
        if statistic is not None and not (isinstance(statistic, float) and math.isnan(statistic)):
            label = _label(str(record["statistic_name"]))
            expected.append((test.name, label, float(statistic), record["statistic_pvalue"]))
        expected.extend(
            (test.name, _label(name), float(value), None) for name, value in test.extras
        )
    assert len(rows) == len(expected), (rows, expected)
    for row, (name, label, value, pvalue) in zip(rows, expected, strict=True):
        assert names[row[0]].name == name, (row, name)
        assert row[1].split(" against ")[0] == label.split(" against ")[0], (row, label)
        assert close(row[2], value), (row, value)
        assert close(row[3], pvalue), (row, pvalue)


def _label(name: str) -> str:
    return vocabulary.STATISTIC_NAMES.get(name, name)


def _read_readings(names: dict[str, Any], labels: list[str], rows: list[list[str]]) -> None:
    assert labels == ["test", "reading", "estimate", "95% CI", "allowed band", "inside"], labels
    expected = [(test, reading) for test in names.values() for reading in test.readings]
    assert len(rows) == len(expected), (rows, expected)
    for row, (test, reading) in zip(rows, expected, strict=True):
        assert names[row[0]] is test, row
        assert row[1] == _label(reading.name), (row, reading.name)
        assert close(row[2], reading.estimate), (row, reading)
        low, high = _numbers(row[3])
        assert close(low, reading.lower) and close(high, reading.upper), (row, reading)
        band_low, band_high = _numbers(row[4])
        assert close(band_low, reading.band[0]) and close(band_high, reading.band[1]), row
        assert row[5] == ("yes" if reading.inside else "no"), (row, reading)
