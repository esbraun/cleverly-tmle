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

__all__ = [
    "header_values",
    "read_corrections",
    "read_equations",
    "read_refutation",
    "sci_close",
    "text_tables",
]

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
        rows: list[list[str]] = []
        for row in lines[index + 1 :]:
            if _rule(row, "=") or not row.strip():
                break
            cells = re.split(r"\s{2,}", row.strip())
            if rows and len(cells) == 1 and len(labels) > 1:
                # A wrapped row: a last column of words continues indented, a row name
                # continues at the margin.
                at = -1 if row.startswith(" ") else 0
                rows[-1][at] = f"{rows[-1][at]} {cells[0]}"
                continue
            rows.append(cells)
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


# ----------------------------------------------------------------- equations and corrections


def sci_close(shown: str, value: float | None, reference: float = float("nan")) -> bool:
    """Whether a number printed in scientific notation agrees with ``value``.

    A printed zero also matches a residue far below ``reference`` (five orders), which the
    summary prints as zero so that it reads the same on every machine; ``"< x"`` matches a
    value at or below ``x``.

    Parameters
    ----------
    shown : str
        The printed number, ``"-"`` for a missing one.
    value : float or None
        The value read from the object.
    reference : float
        The bar the value is read against.

    Returns
    -------
    bool
        Whether the two agree.
    """
    missing = value is None or not math.isfinite(float(value))
    if shown == vocabulary.MISSING:
        return missing
    if missing:
        return False
    number = float(value)  # type: ignore[arg-type]
    if shown.startswith("<"):
        return abs(number) <= float(shown[1:])
    mantissa, _, exponent = shown.partition("e")
    printed = float(shown)
    if printed == 0.0 and math.isfinite(reference) and reference > 0:
        return abs(number) < reference * 1e-5 or abs(number) <= 0.5 * 10 ** -_decimals(mantissa)
    unit = 10 ** (int(exponent or 0) - _decimals(mantissa))
    return abs(printed - number) <= 0.5 * unit * (1 + 1e-9)


def _result_word(passed: bool, threshold: float) -> str:
    if not math.isfinite(threshold):
        return vocabulary.CHECK_RESULTS["descriptive"]
    return vocabulary.CHECK_RESULTS["passed" if passed else "failed"]


def _read_number_cells(cells: list[str], record: dict[str, Any]) -> None:
    error, allowed, ratio, result = cells
    threshold = float(record["threshold"])
    assert sci_close(error, abs(float(record["score"])), threshold), (cells, record)
    assert sci_close(allowed, threshold), (cells, record)
    assert sci_close(ratio, float(record["ratio"]), 1.0), (cells, record)
    assert result == _result_word(bool(record["passed"]), threshold), (cells, record)


def read_equations(check: Any, text: str) -> None:
    """Assert that every number an equation-check summary prints equals the check's.

    Parameters
    ----------
    check : ScoreCheck
        The check.
    text : str
        Its short summary.
    """
    frame = check.to_frame()
    records = [
        dict(zip(frame.columns, values, strict=True)) for values in frame.itertuples(index=False)
    ]
    facts = header_values(text)
    assert (
        facts[vocabulary.RESULT] == vocabulary.CHECK_RESULTS["passed" if check.passed else "failed"]
    )
    assert int(facts[vocabulary.EQUATIONS].replace(",", "")) == len(records)
    assert int(facts[vocabulary.ROWS].replace(",", "")) == check.n
    allowed = facts[vocabulary.ALLOWED_ERROR]
    if check.declared:
        assert check.stopping in allowed, allowed
    else:
        assert close(_numbers(allowed)[0], check.tolerance), allowed

    def recomputed(record: dict[str, Any]) -> bool:
        return record["kind"] in ("correction", "identity") or record["method"] == "recomputed"

    expected = {
        vocabulary.STEP_EQUATIONS_TITLE: [r for r in records if r["kind"] == "fluctuation"],
        vocabulary.CORRECTIONS_TITLE: [r for r in records if recomputed(r)],
        vocabulary.ESTIMATE_EQUATIONS_TITLE: [
            r for r in records if r["kind"] != "fluctuation" and not recomputed(r)
        ],
    }
    tables = text_tables(text)
    assert [title for title, _, _ in tables] == [
        title for title, rows in expected.items() if rows
    ], tables
    for title, labels, rows in tables:
        wanted = expected[title]
        assert len(rows) == len(wanted), (title, rows)
        lead = {
            vocabulary.STEP_EQUATIONS_TITLE: ["equation"],
            vocabulary.ESTIMATE_EQUATIONS_TITLE: ["estimate"],
        }.get(title)
        if lead is None:
            splits = "split" in labels
            lead = ["correction", "arm", *(["split"] if splits else []), "check"]
        assert labels == [*lead, "error", "allowed", "ratio", "result"], labels
        for row, record in zip(rows, wanted, strict=True):
            if title == vocabulary.ESTIMATE_EQUATIONS_TITLE:
                assert row[0] == record["name"], (row, record)
            if title == vocabulary.CORRECTIONS_TITLE:
                arm = re.search(r"\)\[([^\]]+)\]", str(record["name"]))
                assert arm is not None and row[1] == arm.group(1), (row, record)
                draw = re.search(r"\[draw (\d+)\]", str(record["name"]))
                if "split" in labels:
                    assert draw is not None and int(row[2]) == int(draw.group(1)) + 1, row
            _read_number_cells(row[len(lead) :], record)


def read_corrections(check: Any, text: str) -> None:
    """Assert that every number a correction-check summary prints equals the check's.

    Parameters
    ----------
    check : CorrectionCheck
        The check.
    text : str
        Its short summary.
    """
    frame = check.to_frame()
    records = [
        dict(zip(frame.columns, values, strict=True)) for values in frame.itertuples(index=False)
    ]
    facts = header_values(text)
    result = "descriptive" if not records else ("passed" if check.passed else "failed")
    assert facts[vocabulary.RESULT] == vocabulary.CHECK_RESULTS[result]
    assert int(facts[vocabulary.EQUATIONS].replace(",", "")) == sum(
        bool(record["solved"]) for record in records
    )
    assert int(facts[vocabulary.ROWS].replace(",", "")) == check.n
    assert close(_numbers(facts[vocabulary.ALLOWED_ERROR])[0], check.tolerance)
    assert facts[vocabulary.PREDICTIONS] == ("cross-fitted" if check.cross_fitted else "in-sample")
    tables = text_tables(text)
    assert len(tables) == 1, tables
    _, labels, rows = tables[0]
    if not records:
        assert labels == ["corrections"] and rows == [["none reported"]], (labels, rows)
        return
    splits = "split" in labels
    assert splits == (len({record["draw"] for record in records}) > 1)
    lead = ["correction", "arm", *(["split"] if splits else [])]
    assert labels == [*lead, "solved", "in curve", "difference", "cap bias"], labels
    assert len(rows) == len(records), rows
    identity = check.identity_threshold
    for row, record in zip(rows, records, strict=True):
        assert row[1] == str(record["arm"]), (row, record)
        if splits:
            assert int(row[2]) == int(record["draw"]) + 1, (row, record)
        solved, curve, difference, cap = row[len(lead) :]
        bar = check.threshold if record["solved"] else 0.0
        assert sci_close(solved, float(record["stored"]), bar), (row, record)
        assert sci_close(curve, float(record["reported"]), bar), (row, record)
        assert sci_close(difference, float(record["residual"]), identity), (row, record)
        assert sci_close(cap, float(record["clip_bias"]), identity), (row, record)
